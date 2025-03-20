#include <stdio.h>
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>
#include <mutex.h>

#include <alloc.h>
#include <stddef.h>
#include <stdint.h>
#include <barrier.h>
//#include <mram_unaligned.h>


#define T int8_t

// #define ALIGN8(x) (x % 8 == 0 ? x : x / 8 * 8 + 8)

#define TASK_NUM (14)   //14个计算线程， 1个加载线程
#define CONV_SIZE 3

//cout(切了之后)  cin
int16_t cin[13] = {3,64,64,128,128,256,256,256,512,512,512,512,512};
T cout[13] = {64 / 1,64 / 64,128 / 32,128 / 64,256 / 32,256 / 64,256 / 64,
              512 / 32,512 / 64,512 / 64,512 / 16,512 / 16,512 / 16};


T split[13] = {1,64,32,64,32,64,64,
               32,64,64,16,16,16};

//是否需要pooling
T pool[13] = {0,1,0,1,0,0,1,
              0,0,1,0,0,1};
//输出尺寸(=输入)
int16_t out_HW[13] = {224, 224, 112, 112, 56, 56, 56,
                      28, 28, 28, 14, 14, 14};

//out_HW[conv_num] / TASK_NUM
T output_groups[] = {224 / TASK_NUM, 224 / TASK_NUM, 112 / TASK_NUM, 112 / TASK_NUM, 56 / TASK_NUM, 56 / TASK_NUM, 56 / TASK_NUM,
                     28 / TASK_NUM, 28 / TASK_NUM, 28 / TASK_NUM, 14 / TASK_NUM, 14 / TASK_NUM, 14 / TASK_NUM};


//单个缓冲区每次加载的input组数（每层）
T input_groups[13] = {0,2 - 1,3 - 1,11 - 1,17 - 1,25 - 1,25 - 1,16 - 1,24 - 1,24 - 1,0,0,0};

//int input_HXW[13] = {224 * 226, 224 * 226, 112 * 114, 112 * 114, 56 * 58, 56 * 58, 56 * 58,
//                     28 * 30, 28 * 30, 28 * 30, 14 * 16, 14 * 16, 14 * 16};
int output_HXW[13] = {224 * 224, 224 * 224, 112 * 112, 112 * 112, 56 * 56, 56 * 56, 56 * 56,
                      28 * 28, 28 * 28, 28 * 28, 14 * 14, 14 * 14, 14 * 14};


BARRIER_INIT(my_barrier, TASK_NUM + 1);

// #define BUFFER_ALL ((64 - 2) * 1024) //留2K stack？     - 96
//#define BUFFER_IN (64 * 3 * 3 * TASK_NUM) //矩阵加载，input冗余

//WRAM缓存，动态分配
__dma_aligned T* buffer_inA;
__dma_aligned T* buffer_inB;
__dma_aligned T* buffer_weight;
__dma_aligned T* buffer_out;


//MRAM中静态分配，取能用到的最大长度，多给1K
#define INPUT_SIZE  (3200 * 1024)  //((3136 + 1) * 1024) 因为padding，多给点
#define WEIGHT_SIZE ((144 + 1) * 1024)
#define OUTPUT_SIZE ((49 + 1) * 1024)

__mram_noinit T weight[WEIGHT_SIZE];
//双缓冲
__mram_noinit T inputA[INPUT_SIZE];
__mram_noinit T inputB[INPUT_SIZE];

__mram_noinit T outputA[OUTPUT_SIZE];
__mram_noinit T outputB[OUTPUT_SIZE];
T __mram_ptr * inputAB;
T __mram_ptr * outputAB;

#define PAD(H) (H)

T init = 0;
int osize;
T* buffer_in;
T finish = 0;   //=1全部加载完毕
__host __dma_aligned int64_t conv_num,pid;  //所计算的卷积层，和缓冲区


MUTEX_INIT(my_mutex);
T ztb = 0;
// T line[14] = {1,4,7,10,13,2,5,8,11,14,3,6,9,12};
T line[14] = {1,8,5,12,2,9,6,13,3,10,7,14,4,11};

// T line[14] = {1, 8, 11, 4, 13, 6, 9, 2, 12, 5, 10, 3, 7, 14};


//目前计算偏移的消耗过大，改成多维数组能好一点，
//测试了静态数组(多维)，汇编使用移位的方式拼出偏移 ———— 但数组尺寸需要一开始就确定好 ———— 需每层编译一个文件
//测试了T***的形式，比上面的指令更少，———— 但这玩意应该不连续，WRAM传数据开销变大

int main() {

    //最后一号线程负责加载数据
    //从input里面切点太麻烦，下面采用1D conv的形式
    //每个线程负责一行，除了前两行之外，每行可以复用3次（kernel reuse），
    //然后再把output channel 复用了
    //除去weight固定，剩下空间input和output AB双缓冲，output尽量接近2K，算完扔出去
    //根据每层的尺寸情况，开14个线程计算，每次加载14行的整数倍，这种情况下，首尾两行padding不管，注意处理一下
    if(me() == 0){

//        printf("conv-%lld--thread-%d--stack--%d--buffer-%lld\n",conv_num,me(),check_stack(),pid);
        int offset = 0;
        //前9层先装全部的weight和output，size固定，可以存成数组不必计算 //    if(conv_num <= 9)
        if(init == 0){
            init = 1;
            int wsize = cout[conv_num] *
                        cin[conv_num] * CONV_SIZE * CONV_SIZE;
            buffer_weight = mem_alloc(wsize);
            osize = cout[conv_num] *
                    out_HW[conv_num] * out_HW[conv_num];
            buffer_out = mem_alloc(osize);
            //剩下的给input —— 双缓冲
//        int isize = BUFFER_ALL - wsize - osize;
//        buffer_inA = mem_alloc(isize / 2);
//        buffer_inB = mem_alloc(isize / 2);
            int isize = TASK_NUM * out_HW[conv_num] * input_groups[conv_num];
            buffer_inA = mem_alloc(isize);
            buffer_inB = mem_alloc(isize);

//        //读入缓存，实测相比于计算时间，读写可忽略（约百倍差距？道理上不太对），
//        // ———— 但还是overlap吧
//        //加载全部weight
            for(int i = 0;i < (wsize / 2048);i ++,offset += 2048){
                mram_read(weight + offset, buffer_weight + offset, 2048);
//        printf("%d \n",buffer_weight[i + 8]);
            }
            if(wsize % 2048) mram_read(weight + offset, buffer_weight + offset, (wsize % 2048));
        }
        //如果重复launch，全局变量不会清，注意output累加前先清一下
        for(int i = 0;i < osize;i ++){
            buffer_out[i] = 0;
        }

        if(pid % 2){
            inputAB = inputA;
            outputAB = outputA;
        }else{
            inputAB = inputB;
            outputAB = outputB;
        }
//        //循环加载input缓冲A或B,14行一组
//        //一张图每次拿14行
        T* buffer_inAB = buffer_inA;
        int finish_line = 0,write_offset = 0;
        for (int z = 0; z < out_HW[conv_num]; z += TASK_NUM) {
            offset = z * PAD(out_HW[conv_num]);
            //输入通道方向每次拿input_groups[conv_num]组
            for(int t = 0;t < cin[conv_num] / input_groups[conv_num];t ++){
                //每次read 14行
                int buffer_offset = 0;
                for(int b = 0;b < input_groups[conv_num];b ++,offset += output_HXW[conv_num]){
//                    printf("%d %d %d\n",z,t,b);
                    //14行
                    int size = TASK_NUM * PAD(out_HW[conv_num]);

                    if(size > 2048){
                        //根据实际大小，最多传两次就行，不写循环了
                        mram_read(inputAB + offset, buffer_inAB + buffer_offset, 2048);
                        mram_read(inputAB + offset + 2048, buffer_inAB + buffer_offset + 2048, (size - 2048));
                    }else{
                        mram_read(inputAB + offset, buffer_inAB + buffer_offset, size);
                    }

                    buffer_offset += size;
                }
                buffer_in = buffer_inAB;    //给计算线程
                if(buffer_inAB == buffer_inA){
                    buffer_inAB = buffer_inB;
                }else{
                    buffer_inAB = buffer_inA;
                }
                //完成一次缓冲区加载，唤醒计算线程
                barrier_wait(&my_barrier);
            }


            //剩下的通道
//            if(cin[conv_num] % input_groups[conv_num]){
////                for
//            }
//            barrier_wait(&my_barrier);

            if(finish_line){
                //一组（14行）的所有通道计算完成，写入MRAM（第一组13行、中间14行）----由于对齐原因，改成第一组写12行，中间14，最后16
                int size = finish_line * out_HW[conv_num];
                //输出通道
                for (int i = 0; i < cout[conv_num]; ++i) {

                    int cout_offset = i * output_HXW[conv_num];
//                    printf("%d %d\n",write_offset,cout_offset);

                    if(size > 2048){
                        mram_write(buffer_out + write_offset + cout_offset, outputAB + write_offset + cout_offset, 2048);
                        mram_write(buffer_out + write_offset + cout_offset + 2048, outputAB + write_offset + cout_offset + 2048, (size - 2048));
                    }else{
                        mram_write(buffer_out + write_offset + cout_offset, outputAB + write_offset + cout_offset, (size));
                    }

                }

                write_offset += size;
                finish_line = 14;
            } else{
                write_offset = 0;
                finish_line = 12;
            }


        }

        //全部加载完毕，等最后一组output算完
        finish = 1;
        barrier_wait(&my_barrier);
        //传回MRAM（最后一组15行）----16行
        finish_line = 16;
        int size = finish_line * out_HW[conv_num];
        //输出通道
        for (int i = 0; i < cout[conv_num]; ++i) {

            int cout_offset = i * output_HXW[conv_num];
//            printf("%d %d\n",write_offset,cout_offset);

            if(size > 2048){
                mram_write(buffer_out + write_offset + cout_offset, outputAB + write_offset + cout_offset, 2048);
                mram_write(buffer_out + write_offset + cout_offset + 2048, outputAB + write_offset + cout_offset + 2048, (size - 2048));
            }else{
                mram_write(buffer_out + write_offset + cout_offset, outputAB + write_offset + cout_offset, (size));
            }

        }
        //最后走关灯
        finish = 0;


    }else{

    //    printf("conv-%lld--thread-%d--stack--%d--buffer-%lld\n",conv_num,me(),check_stack(),pid);

        //----之后想办法把里面的乘法去掉，或者改成8位乘法

        //循环拿缓冲A或B的数据，算卷积
        int16_t cin_ = 0,wsize_cout = cin[conv_num] * 9;
        T groups = 0,first = 0,last = 0; //当前组数，是否是第一行、最后一行
        while(finish != 1) {
            //等缓冲区加载完
            barrier_wait(&my_barrier);

            mutex_lock(my_mutex);
            T myline = line[ztb];
            ztb ++;
            if(ztb == 14) ztb = 0;
            mutex_unlock(my_mutex);

            // T myline = me();

            //循环每次处理一个cin（上的14行（14线程））
            for(T b = 0;b < input_groups[conv_num];b ++){
                //本线程负责的行
                T* in = buffer_in + (b * TASK_NUM + myline - 1) * PAD(out_HW[conv_num]);

                if(myline - 1 == 0 && groups == 0) first = 1;
                else first = 0;
                if(myline - 1 == 13 && groups == output_groups[conv_num] - 1) last = 1;
                else last = 0;

                //比较：1D conv 基本可以完全reuse kernel(横向和纵向)和cout —— 寄存器不够，cout可能没法完全复用；
                //而矩阵方式只能reuse cout 和一部分横向kernel（每个线程算相邻几个点）
                //1D conv相对的缺点：有锁的风险，以及if能通过压数据变成向量内积，相比于矩阵方式可能浪费一些空间（用不满寄存器）

                // 1D conv，buffer_out是psum，需要cin和3行kernel的结果累加
                for(int16_t i = 0;i < out_HW[conv_num];i += 2){
                    //所有cout的weight能固定在寄存器最好，目前寄存器不够
                    for(T j = 0;j < cout[conv_num];j ++){

                        T* wei = buffer_weight + j * wsize_cout + cin_ * 9;
                        T* out = buffer_out + j * output_HXW[conv_num] + (groups * TASK_NUM + myline - 1) * out_HW[conv_num];

                        mutex_lock(my_mutex);
                        // if(i == 0){

                        //     if(!last) out[i + out_HW[conv_num]] += (wei[1] * in[i]
                        //                                             + wei[2] * in[i + 1]);
                        //     out[i] +=        (wei[4] * in[i]
                        //                       + wei[5] * in[i + 1]);
                        //     if(!first) out[i - out_HW[conv_num]] += (wei[7] * in[i]
                        //                                              + wei[8] * in[i + 1]);

                        // } else if(i == out_HW[conv_num] - 1){

                        //     if(!last) out[i + out_HW[conv_num]] += (wei[0] * in[i - 1]
                        //                                             + wei[1] * in[i]);
                        //     out[i] +=        (wei[3] * in[i - 1]
                        //                       + wei[4] * in[i]);
                        //     if(!first) out[i - out_HW[conv_num]] += (wei[6] * in[i - 1]
                        //                                              + wei[7] * in[i]);

                        // } else{

                        //     if(!last) out[i + out_HW[conv_num]] += (wei[0] * in[i - 1]
                        //                                             + wei[1] * in[i]
                        //                                             + wei[2] * in[i + 1]);
                        //     out[i] +=        (wei[3] * in[i - 1]
                        //                       + wei[4] * in[i]
                        //                       + wei[5] * in[i + 1]);
                        //     if(!first) out[i - out_HW[conv_num]] += (wei[6] * in[i - 1]
                        //                                              + wei[7] * in[i]
                        //                                              + wei[8] * in[i + 1]);

                        // }


                        if(!last){
                            if(i == 0){
                                out[i + out_HW[conv_num]] += (wei[1] * in[i]
                                                                    + wei[2] * in[i + 1]);
                            }else{
                                out[i + out_HW[conv_num]] += (wei[0] * in[i - 1]
                                                                    + wei[1] * in[i]
                                                                    + wei[2] * in[i + 1]);
                            }

                            if(i == out_HW[conv_num] - 2){
                                out[i + out_HW[conv_num] + 1] += (wei[0] * in[i]
                                                                    + wei[1] * in[i + 1]);
                            }else{
                                out[i + out_HW[conv_num] + 1] += (wei[0] * in[i]
                                                                    + wei[1] * in[i + 1]
                                                                    + wei[2] * in[i + 2]);
                            }
                                
                        }

                        if(i == 0){
                            out[i] +=        (wei[4] * in[i]
                                              + wei[5] * in[i + 1]);
                        }else{
                            out[i] +=        (wei[3] * in[i - 1]
                                              + wei[4] * in[i]
                                              + wei[5] * in[i + 1]);
                        }
                        
                        if(i == out_HW[conv_num] - 2){
                            out[i + 1] +=    (wei[3] * in[i]
                                              + wei[4] * in[i + 1]);
                        }else{
                            out[i + 1] +=    (wei[3] * in[i]
                                              + wei[4] * in[i + 1]
                                              + wei[5] * in[i + 2]);
                        }
                            
                        if(!first){
                            if(i == 0){
                                out[i - out_HW[conv_num]] += (wei[7] * in[i]
                                                                     + wei[8] * in[i + 1]);
                            }else{
                                out[i - out_HW[conv_num]] += (wei[6] * in[i - 1]
                                                                     + wei[7] * in[i]
                                                                     + wei[8] * in[i + 1]);
                            }
                            
                            if(i == out_HW[conv_num] - 2){
                                out[i - out_HW[conv_num] + 1] += (wei[6] * in[i]
                                                                     + wei[7] * in[i + 1]);
                            }else{
                                out[i - out_HW[conv_num] + 1] += (wei[6] * in[i]
                                                                     + wei[7] * in[i + 1]
                                                                     + wei[8] * in[i + 2]);
                            }
                                
                        } 
                        mutex_unlock(my_mutex);

                    } 

                }

                cin_ ++;
                if(cin_ == cin[conv_num]){
                    cin_ = 0;
                    groups ++;
                }
            }


        }
        //所有人把最后一组output算完，通知加载线程传回
        barrier_wait(&my_barrier);

    }

    return 0;
}


//void scan(T *in,int C,int H,int W){
//
//    printf("scan\n");
//    for (int i = 0; i < C; ++i)
//    {
//        for (int j = 0; j < H; ++j)
//        {
//            for (int k = 0; k < W; ++k)
//                printf("%d\t",in[i * H * W + j * W + k]);
//            printf("\n");
//        }
//        printf("\n");
//    }
//
//    printf("\n");
//}
//
//void clear_buffer(T * in,int C,int H,int W){
//
//    printf("clear\n");
//    for (int i = 0; i < C; ++i)
//        for (int j = 0; j < H; ++j)
//            for (int k = 0; k < W; ++k)
//                in[i * H * W + j * W + k] = 0;
//
//}
//
////这个开销太大，通过传输时的位置来做padding —— 算了，直接cpu做吧
//T* padding(T* in,T* out,int ch,int H,int W){
//
//    clear_buffer(out,ch,H + 2,W + 2);
//
//    for(int i = 0;i < ch;i ++)
//        for(int j = 0;j < H;j ++)
//            for(int k = 0;k < W;k ++)
//                out[i * (H + 2) * (W + 2) + (j + 1) * (W + 2) + (k + 1)]
//                        = in[i * H * W + j * W + k];
//    return out;
//}
//
//void conv(T* in,T* weights,T* out,
//          int Tr, int Tc, int Tm, int Tn, int K){
//
//    clear_buffer(out,Tm,Tr,Tc);
//
//    for(int r = 0; r < Tr; r++)
//        for(int c = 0; c < Tc; c++)
//            for(int i = 0; i < K; i++)
//                for(int j = 0; j < K; j++)
//                {
//                    for(int tm = 0; tm < Tm; tm++)  //UNROLL
//// if(n == 0)conv.out[m + tm][r][c] = conv.bias[m + tm];
//                        for(int tn = 0; tn < Tn; tn++)  //UNROLL
//                        {
//                            out[(tm) * Tr * Tc + r * Tc + c] +=
//                                    weights[(tm) * Tn * K * K + (tn) * K * K + i * K + j] *
//                                    in[(tn) * ((Tr - 1) + K) * ((Tc - 1) + K) +
//                                       (r + i) * ((Tc - 1) + K) + (c + j)];
//// if(i == K - 1 && j == K - 1 && conv[idx].out[m + tm][r][c] < 0)conv[idx].out[m + tm][r][c] = 0;     //ReLU
//                        }
//                }
//
//}
//
//T max_of_4(T a, T b, T c, T d) {
//    if (a >= b && a >= c && a >= d) return a;
//    if (b >= c && b >= d) return b;
//    if (c >= d) return c;
//    return d;
//}
//
//T* maxpooling(T* out,int ch,int H,int W) {
//
//    for(int i = 0;i < ch;i ++)
//        for(int j = 0;j < H;j += 2)
//            for(int k = 0;k < W;k += 2)
//                out[i * (H / 2) * (W / 2) + (j / 2) * (W / 2) + (k / 2)]
//                        = max_of_4(out[i * H * W + j * W + k], out[i * H * W + (j + 1) * W + k],
//                                   out[i * H * W + j * W + (k + 1)], out[i * H * W + (j + 1) * W + (k + 1)]);
//    return out;
//}