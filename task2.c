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

// 宏定义会导致错误，貌似编译器问题？之后再解决
#define CIN 64
#define COUT 1
#define OUTHW 224
#define IN_GROUPS 1
#define OUTHXW 50176
#define OUT_GROUPS 16
// 多线程问题估计只能通过把output空间翻3倍来解决，也之后再说


// #define CIN cin[conv_num]
// #define COUT cout[conv_num]
// #define OUTHW out_HW[conv_num]
// #define IN_GROUPS input_groups[conv_num]
// #define OUTHXW output_HXW[conv_num]
// #define OUT_GROUPS output_groups[conv_num]


//是否需要pooling
T pool[13] = {0,1,0,1,0,0,1,
              0,0,1,0,0,1};

int16_t cin[13] = {3,64,64,128,128,256,256,256,512,512,512,512,512};
T cout[13] = {64 / 1,64 / 64,128 / 32,128 / 64,256 / 32,256 / 64,256 / 64,
             512 / 32,512 / 64,512 / 64,512 / 16,512 / 16,512 / 16};

//输出尺寸(=输入)
int16_t out_HW[13] = {224, 224, 112, 112, 56, 56, 56,
                      28, 28, 28, 14, 14, 14};

//OUTHW / TASK_NUM
T output_groups[] = {224 / TASK_NUM, 224 / TASK_NUM, 112 / TASK_NUM, 112 / TASK_NUM, 56 / TASK_NUM, 56 / TASK_NUM, 56 / TASK_NUM,
                     28 / TASK_NUM, 28 / TASK_NUM, 28 / TASK_NUM, 14 / TASK_NUM, 14 / TASK_NUM, 14 / TASK_NUM};


//单个缓冲区每次加载的input组数（每层）
T input_groups[13] = {0,2 - 1,3 - 1,11 - 1,17 - 1,25 - 1,25 - 1,16 - 1,24 - 1,24 - 1,0,0,0};

int output_HXW[13] = {224 * 224, 224 * 224, 112 * 112, 112 * 112, 56 * 56, 56 * 56, 56 * 56,
                     28 * 28, 28 * 28, 28 * 28, 14 * 14, 14 * 14, 14 * 14};


BARRIER_INIT(my_barrier, TASK_NUM + 1);

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

extern void conv_1d(T* in,T* out,int16_t size,T* wei);

// void conv_1d(T* in,T* out,int16_t size,T* wei){

//                         for(int16_t i = 0;i < size;i ++){  

//                             if(i == 0){

//                                 out[i + size] += (wei[1] * in[i]
//                                                 + wei[2] * in[i + 1]);
//                                 out[i] +=        (wei[4] * in[i]
//                                                 + wei[5] * in[i + 1]);
//                                 out[i - size] += (wei[7] * in[i]
//                                                 + wei[8] * in[i + 1]);

//                             } else if(i == size - 1){

//                                 out[i + size] += (wei[0] * in[i - 1]
//                                                 + wei[1] * in[i]);
//                                 out[i] +=        (wei[3] * in[i - 1]
//                                                 + wei[4] * in[i]);
//                                 out[i - size] += (wei[6] * in[i - 1]
//                                                 + wei[7] * in[i]);

//                             } else{

//                                 out[i + size] += (wei[0] * in[i - 1]
//                                                 + wei[1] * in[i]
//                                                 + wei[2] * in[i + 1]);
//                                 out[i] +=        (wei[3] * in[i - 1]
//                                                 + wei[4] * in[i]
//                                                 + wei[5] * in[i + 1]);
//                                 out[i - size] += (wei[6] * in[i - 1]
//                                                 + wei[7] * in[i]
//                                                 + wei[8] * in[i + 1]);

//                             }

//                         }

// }


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

    //    printf("conv-%lld--thread-%d--stack--%d--buffer-%lld\n",conv_num,me(),check_stack(),pid);
        int offset = 0;
        //前9层先装全部的weight和output，size固定，可以存成数组不必计算 //    if(conv_num <= 9)
        if(init == 0){
            init = 1;
            int wsize = COUT *
                        CIN * CONV_SIZE * CONV_SIZE;
            buffer_weight = mem_alloc(wsize);
            osize = COUT *
            OUTHW * OUTHW;
            buffer_out = mem_alloc(osize);
            //剩下的给input —— 双缓冲
            int isize = TASK_NUM * OUTHW * IN_GROUPS;
            buffer_inA = mem_alloc(isize);
            buffer_inB = mem_alloc(isize);

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
        T finish_line = 0;
        int write_offset = 0;
        for (int16_t z = 0; z < OUTHW; z += TASK_NUM) {
            offset = z * PAD(OUTHW);
            //输入通道方向每次拿IN_GROUPS组
            for(int16_t t = 0;t < CIN / IN_GROUPS;t ++){
                //每次read 14行
                int buffer_offset = 0;
                for(T b = 0;b < IN_GROUPS;b ++,offset += OUTHXW){
//                    printf("%d %d %d\n",z,t,b);
                    //14行
                    int16_t size = TASK_NUM * PAD(OUTHW);

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
//            if(CIN % IN_GROUPS){
////                for
//            }
//            barrier_wait(&my_barrier);

            if(finish_line){
                //一组（14行）的所有通道计算完成，写入MRAM（第一组13行、中间14行）----由于对齐原因，改成第一组写12行，中间14，最后16
                int size = finish_line * OUTHW;
                //输出通道
                for (int i = 0; i < COUT; ++i) {

                    int cout_offset = i * OUTHXW;
                    
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
        int size = finish_line * OUTHW;
        //输出通道
        for (T i = 0; i < COUT; ++i) {

            int cout_offset = i * OUTHXW;
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
        int16_t cin_ = 0,wsize_cout = CIN * 9;
        T groups = 0,first = 0,last = 0; //当前组数，是否是第一行、最后一行
        while(finish != 1) {
            //等缓冲区加载完
            barrier_wait(&my_barrier);

            // mutex_lock(my_mutex);
            // T myline = line[ztb];
            // ztb ++;
            // if(ztb == 14) ztb = 0;
            // mutex_unlock(my_mutex);

            T myline = me();

            //循环每次处理一个cin（上的14行（14线程））
            for(T b = 0;b < IN_GROUPS;b ++){
                //本线程负责的行
                T* in = buffer_in + (b * TASK_NUM + myline - 1) * OUTHW;

                if(myline - 1 == 0 && groups == 0) first = 1;
                else first = 0;
                if(myline - 1 == 13 && groups == output_groups[conv_num] - 1) last = 1;
                else last = 0;


                //在寄存器层面，可以测试不同stationary的差异
                //但由于寄存器数量有限，不能完全测试
                //1~6层 w > o > i   w没法全放下，固定一个cout，保证先用完
                //7~9   o > w > i
                //10~12 o > i > w
                if(first){

                    for(T j = 0;j < COUT;j ++){

                        T* wei = buffer_weight + (j * CIN + cin_) * 9;
                        T* out = buffer_out + j * OUTHXW + (groups * TASK_NUM + myline - 1) * OUTHW;

                        for(int16_t i = 0;i < OUTHW;i += 2){  
                        
                            // mutex_lock(my_mutex);
                            if(i == 0){
                                out[i + OUTHW] += (wei[1] * in[i]
                                                              + wei[2] * in[i + 1]);
                            }else{
                                out[i + OUTHW] += (wei[0] * in[i - 1]
                                                              + wei[1] * in[i]
                                                              + wei[2] * in[i + 1]);
                            }

                            if(i == OUTHW - 2){
                                out[i + OUTHW + 1] += (wei[0] * in[i]
                                                                  + wei[1] * in[i + 1]);
                            }else{
                                out[i + OUTHW + 1] += (wei[0] * in[i]
                                                                  + wei[1] * in[i + 1]
                                                                  + wei[2] * in[i + 2]);
                            }

                            //
                            if(i == 0){
                                out[i] +=        (wei[4] * in[i]
                                                + wei[5] * in[i + 1]);
                            }else{
                                out[i] +=        (wei[3] * in[i - 1]
                                                + wei[4] * in[i]
                                                + wei[5] * in[i + 1]);
                            }

                            if(i == OUTHW - 2){
                                out[i + 1] +=    (wei[3] * in[i]
                                                + wei[4] * in[i + 1]);
                            }else{
                                out[i + 1] +=    (wei[3] * in[i]
                                                + wei[4] * in[i + 1]
                                                + wei[5] * in[i + 2]);
                            }
                            // mutex_unlock(my_mutex);

                        }

                    }
                }else if(last){

                    for(T j = 0;j < COUT;j ++){

                        T* wei = buffer_weight + (j * CIN + cin_) * 9;
                        T* out = buffer_out + j * OUTHXW + (groups * TASK_NUM + myline - 1) * OUTHW;

                        for(int16_t i = 0;i < OUTHW;i += 2){  

                            // mutex_lock(my_mutex);
                            if(i == 0){
                                out[i] +=        (wei[4] * in[i]
                                                + wei[5] * in[i + 1]);
                            }else{
                                out[i] +=        (wei[3] * in[i - 1]
                                                + wei[4] * in[i]
                                                + wei[5] * in[i + 1]);
                            }

                            if(i == OUTHW - 2){
                                out[i + 1] +=    (wei[3] * in[i]
                                                + wei[4] * in[i + 1]);
                            }else{
                                out[i + 1] +=    (wei[3] * in[i]
                                                + wei[4] * in[i + 1]
                                                + wei[5] * in[i + 2]);
                            }

                            //
                            if(i == 0){
                                out[i - OUTHW] += (wei[7] * in[i]
                                                              + wei[8] * in[i + 1]);
                            }else{
                                out[i - OUTHW] += (wei[6] * in[i - 1]
                                                              + wei[7] * in[i]
                                                              + wei[8] * in[i + 1]);
                            }

                            if(i == OUTHW - 2){
                                out[i - OUTHW + 1] += (wei[6] * in[i]
                                                                  + wei[7] * in[i + 1]);
                            }else{
                                out[i - OUTHW + 1] += (wei[6] * in[i]
                                                                  + wei[7] * in[i + 1]
                                                                  + wei[8] * in[i + 2]);
                            }
                            // mutex_unlock(my_mutex);

                        }

                    }

                }else{


                    for(T j = 0;j < COUT;j ++){

                        T* wei = buffer_weight + (j * CIN + cin_) * 9;
                        T* out = buffer_out + j * OUTHXW + (groups * TASK_NUM + myline - 1) * OUTHW;

                        // mutex_lock(my_mutex);

                        conv_1d(in,out,OUTHW,wei);

                        // mutex_unlock(my_mutex);

                    }

                }

                cin_ ++;
                if(cin_ == CIN){
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




            //   for(T b = 0;b < IN_GROUPS;b ++){
            //     //本线程负责的行
            //     T* in = buffer_in + (b * TASK_NUM + myline - 1) * OUTHW;

            //     if(myline - 1 == 0 && groups == 0) first = 1;
            //     else first = 0;
            //     if(myline - 1 == 13 && groups == output_groups[conv_num] - 1) last = 1;
            //     else last = 0;


            //     for(int16_t i = 0;i < OUTHW;i += 2){
            //         //所有cout的weight能固定在寄存器最好，目前寄存器不够
            //         for(T j = 0;j < COUT;j ++){

            //             T* wei = buffer_weight + (j * CIN + cin_) * 9;
            //             T* out = buffer_out + j * OUTHXW + (groups * TASK_NUM + myline - 1) * OUTHW;

            //             if(!last){
            //                 if(i == 0){
            //                     out[i + OUTHW] += (wei[1] * in[i]
            //                                                   + wei[2] * in[i + 1]);
            //                 }else{
            //                     out[i + OUTHW] += (wei[0] * in[i - 1]
            //                                                   + wei[1] * in[i]
            //                                                   + wei[2] * in[i + 1]);
            //                 }

            //                 if(i == OUTHW - 1){
            //                     out[i + OUTHW + 1] += (wei[0] * in[i]
            //                                                       + wei[1] * in[i + 1]);
            //                 }else{
            //                     out[i + OUTHW + 1] += (wei[0] * in[i]
            //                                                       + wei[1] * in[i + 1]
            //                                                       + wei[2] * in[i + 2]);
            //                 }

            //             }

            //             if(i == 0){
            //                 out[i] +=        (wei[4] * in[i]
            //                                   + wei[5] * in[i + 1]);
            //             }else{
            //                 out[i] +=        (wei[3] * in[i - 1]
            //                                   + wei[4] * in[i]
            //                                   + wei[5] * in[i + 1]);
            //             }

            //             if(i == OUTHW - 1){
            //                 out[i + 1] +=    (wei[3] * in[i]
            //                                   + wei[4] * in[i + 1]);
            //             }else{
            //                 out[i + 1] +=    (wei[3] * in[i]
            //                                   + wei[4] * in[i + 1]
            //                                   + wei[5] * in[i + 2]);
            //             }

            //             if(!first){
            //                 if(i == 0){
            //                     out[i - OUTHW] += (wei[7] * in[i]
            //                                                   + wei[8] * in[i + 1]);
            //                 }else{
            //                     out[i - OUTHW] += (wei[6] * in[i - 1]
            //                                                   + wei[7] * in[i]
            //                                                   + wei[8] * in[i + 1]);
            //                 }

            //                 if(i == OUTHW - 1){
            //                     out[i - OUTHW + 1] += (wei[6] * in[i]
            //                                                       + wei[7] * in[i + 1]);
            //                 }else{
            //                     out[i - OUTHW + 1] += (wei[6] * in[i]
            //                                                       + wei[7] * in[i + 1]
            //                                                       + wei[8] * in[i + 2]);
            //                 }

            //             }

            //         }

            //     }

            //     cin_ ++;
            //     if(cin_ == CIN){
            //         cin_ = 0;
            //         groups ++;
            //     }
            // }


                // if(first){

                //     for(T j = 0;j < COUT;j ++){

                //         T* wei = buffer_weight + (j * CIN + cin_) * 9;
                //         T* out = buffer_out + j * OUTHXW + (groups * TASK_NUM + myline - 1) * OUTHW;

                //         for(int16_t i = 0;i < OUTHW;i += 2){  
                        
                //             // mutex_lock(my_mutex);
                //             if(i == 0){
                //                 out[i + OUTHW] += (wei[1] * in[i]
                //                                               + wei[2] * in[i + 1]);
                //             }else{
                //                 out[i + OUTHW] += (wei[0] * in[i - 1]
                //                                               + wei[1] * in[i]
                //                                               + wei[2] * in[i + 1]);
                //             }

                //             if(i == OUTHW - 2){
                //                 out[i + OUTHW + 1] += (wei[0] * in[i]
                //                                                   + wei[1] * in[i + 1]);
                //             }else{
                //                 out[i + OUTHW + 1] += (wei[0] * in[i]
                //                                                   + wei[1] * in[i + 1]
                //                                                   + wei[2] * in[i + 2]);
                //             }

                //             //
                //             if(i == 0){
                //                 out[i] +=        (wei[4] * in[i]
                //                                 + wei[5] * in[i + 1]);
                //             }else{
                //                 out[i] +=        (wei[3] * in[i - 1]
                //                                 + wei[4] * in[i]
                //                                 + wei[5] * in[i + 1]);
                //             }

                //             if(i == OUTHW - 2){
                //                 out[i + 1] +=    (wei[3] * in[i]
                //                                 + wei[4] * in[i + 1]);
                //             }else{
                //                 out[i + 1] +=    (wei[3] * in[i]
                //                                 + wei[4] * in[i + 1]
                //                                 + wei[5] * in[i + 2]);
                //             }
                //             // mutex_unlock(my_mutex);

                //         }

                //     }
                // }else if(last){

                //     for(T j = 0;j < COUT;j ++){

                //         T* wei = buffer_weight + (j * CIN + cin_) * 9;
                //         T* out = buffer_out + j * OUTHXW + (groups * TASK_NUM + myline - 1) * OUTHW;

                //         for(int16_t i = 0;i < OUTHW;i += 2){  

                //             // mutex_lock(my_mutex);
                //             if(i == 0){
                //                 out[i] +=        (wei[4] * in[i]
                //                                 + wei[5] * in[i + 1]);
                //             }else{
                //                 out[i] +=        (wei[3] * in[i - 1]
                //                                 + wei[4] * in[i]
                //                                 + wei[5] * in[i + 1]);
                //             }

                //             if(i == OUTHW - 2){
                //                 out[i + 1] +=    (wei[3] * in[i]
                //                                 + wei[4] * in[i + 1]);
                //             }else{
                //                 out[i + 1] +=    (wei[3] * in[i]
                //                                 + wei[4] * in[i + 1]
                //                                 + wei[5] * in[i + 2]);
                //             }

                //             //
                //             if(i == 0){
                //                 out[i - OUTHW] += (wei[7] * in[i]
                //                                               + wei[8] * in[i + 1]);
                //             }else{
                //                 out[i - OUTHW] += (wei[6] * in[i - 1]
                //                                               + wei[7] * in[i]
                //                                               + wei[8] * in[i + 1]);
                //             }

                //             if(i == OUTHW - 2){
                //                 out[i - OUTHW + 1] += (wei[6] * in[i]
                //                                                   + wei[7] * in[i + 1]);
                //             }else{
                //                 out[i - OUTHW + 1] += (wei[6] * in[i]
                //                                                   + wei[7] * in[i + 1]
                //                                                   + wei[8] * in[i + 2]);
                //             }
                //             // mutex_unlock(my_mutex);

                //         }

                //     }

                // }else{


                //     for(T j = 0;j < COUT;j ++){

                //         T* wei = buffer_weight + (j * CIN + cin_) * 9;
                //         T* out = buffer_out + j * OUTHXW + (groups * TASK_NUM + myline - 1) * OUTHW;

                //         for(int16_t i = 0;i < OUTHW;i += 2){  

                //             // mutex_lock(my_mutex);
                //             if(i == 0){
                //                 out[i + OUTHW] += (wei[1] * in[i]
                //                                               + wei[2] * in[i + 1]);
                //             }else{
                //                 out[i + OUTHW] += (wei[0] * in[i - 1]
                //                                               + wei[1] * in[i]
                //                                               + wei[2] * in[i + 1]);
                //             }

                //             if(i == OUTHW - 2){
                //                 out[i + OUTHW + 1] += (wei[0] * in[i]
                //                                                   + wei[1] * in[i + 1]);
                //             }else{
                //                 out[i + OUTHW + 1] += (wei[0] * in[i]
                //                                                   + wei[1] * in[i + 1]
                //                                                   + wei[2] * in[i + 2]);
                //             }

                //             //
                //             if(i == 0){
                //                 out[i] +=        (wei[4] * in[i]
                //                                 + wei[5] * in[i + 1]);
                //             }else{
                //                 out[i] +=        (wei[3] * in[i - 1]
                //                                 + wei[4] * in[i]
                //                                 + wei[5] * in[i + 1]);
                //             }

                //             if(i == OUTHW - 2){
                //                 out[i + 1] +=    (wei[3] * in[i]
                //                                 + wei[4] * in[i + 1]);
                //             }else{
                //                 out[i + 1] +=    (wei[3] * in[i]
                //                                 + wei[4] * in[i + 1]
                //                                 + wei[5] * in[i + 2]);
                //             }

                //             //
                //             if(i == 0){
                //                 out[i - OUTHW] += (wei[7] * in[i]
                //                                               + wei[8] * in[i + 1]);
                //             }else{
                //                 out[i - OUTHW] += (wei[6] * in[i - 1]
                //                                               + wei[7] * in[i]
                //                                               + wei[8] * in[i + 1]);
                //             }

                //             if(i == OUTHW - 2){
                //                 out[i - OUTHW + 1] += (wei[6] * in[i]
                //                                                   + wei[7] * in[i + 1]);
                //             }else{
                //                 out[i - OUTHW + 1] += (wei[6] * in[i]
                //                                                   + wei[7] * in[i + 1]
                //                                                   + wei[8] * in[i + 2]);
                //             }

                //             // mutex_unlock(my_mutex);

                //         }

                //     }

                // }