#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <alloc.h>
#include <stddef.h>


#define T int8_t

//#define C_in 3
//#define C_out 64
//
//#define H_ 224
//#define W_ 224
//#define K_ 3
//#define S_ 1
//
//#define ALIGN8(x) x % 8 == 0 ? x : x / 8 * 8 + 8
//
//#define out_size  6
//
//#define IMG_SIZE 224
//#define CONV_SIZE 3
//#define DPU_NUM 8


__dma_aligned T* buffer1;
__dma_aligned T* buffer_weight;
__dma_aligned T* buffer2;
__dma_aligned T buffer[3][3][224];

//__mram_noinit T input[3 * 224 * 224];
__mram_noinit T input[3][224][224];
//__mram_noinit T weight[64 * 3 * 3 * 3];


#define SIZE (64 * 3 * 3)

int main() {

//    printf("hello~\n");

//    buddy_init(60000);

//    buffer1 = buddy_alloc(10000);
//    buffer_weight = buddy_alloc(10000);
//    buffer2 = buddy_alloc(10000);

//    for(int i = 0;i < 10000;i ++){
//        printf("%d\t",data[i]);
//        if(i % 99 == 0) printf("\n");
//    }
//    buddy_free(data);


    buffer_weight = mem_alloc(SIZE);
//    mram_read(weight, buffer_weight, SIZE * sizeof(T));
    buffer1 = mem_alloc(SIZE);
    buffer2 = mem_alloc(SIZE * 3);
    mram_read(input, buffer1, SIZE * sizeof(T));

    int a = 3,b = 3,c = 224;
    T*** image = mem_alloc(a * sizeof(T**));
    for (int i = 0; i < a; i++) {
        image[i] = mem_alloc(b * sizeof(T*));
        for (int j = 0; j < b; j++) {
            image[i][j] = mem_alloc(c * sizeof(T));
        }
    }

    int i = 0;
    T* buf = &image[0][0][0];
    for(;i < 3 * 3 * 224;i ++){
        buf[i] = i;
//        buffer1[i] = i;
//        printf("%d\t",buffer1[i]);
    }

    for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 224; ++l) {
                image[0][0][0] += image[j][k][l];
            }
        }

    }

//比较：1D conv 基本可以完全reuse kernel(横向和纵向)和cout —— 寄存器不够，cout可能没法完全复用；
//而矩阵方式只能reuse cout 和一部分横向kernel（每个线程算相邻几个点）
//1D conv相对的缺点：有锁的风险，以及if能通过压数据变成向量内积，相比于矩阵方式可能浪费一些空间（用不满寄存器）

// 1D conv，buffer2是psum，需要cin和3行kernel的结果累加
//    register T w0 = buffer_weight[0];
//    for(i = 0;i < 224;i ++){
//        buffer2[i] += (buffer_weight[0] * buffer1[i]
//                       + buffer_weight[1] * buffer1[i + 1]
//                       + buffer_weight[2] * buffer1[i + 2]);
////        buffer2[i + 224] += (buffer_weight[3] * buffer1[i]
////                       + buffer_weight[4] * buffer1[i + 1]
////                       + buffer_weight[5] * buffer1[i + 2]);
////        buffer2[i + 224 + 224] += (buffer_weight[6] * buffer1[i]
////                       + buffer_weight[7] * buffer1[i + 1]
////                       + buffer_weight[8] * buffer1[i + 2]);
//    }

// 向量内积
//    for(i = 0;i < SIZE;i ++){
//        T t1 = ((T)buffer_weight[i]);
//        T t2 = ((T)buffer1[i]);
//        buffer2[i] += ((t1 * t2)
////                       + buffer_weight[i] * buffer1[i]
////                       + buffer_weight[i] * buffer1[i]
////                       + buffer_weight[i] * buffer1[i]
//                       );
//    }

    
//    for(int r = 0; r < 224; r += ){
//
//        for(int c = 0; c < 224; c += ){
//
//            mram_read(input, buffer1, 3 * 224 * 224 * sizeof(T));
//
//            conv(buffer1,buffer_weight,buffer2,int Tr, int Tc, 64, 3, 3);
//        }
//
//    }

    return 0;
}




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
//
//T* padding(T* in,T* out,int ch,int H,int W){
//
//    clear_buffer(out,ch,H + 2,W + 2);
//
//    for(int i = 0;i < ch;i ++)
//        for(int j = 0;j < H;j ++)
//            for(int k = 0;k < W;k ++)
//                out[i * (H + 2) * (W + 2) + (j + 1) * (W + 2) + (k + 1)]
//                        = in[i * H * W + j * W + k];
//
//    return out;
//}
//
//void conv(T *in,T *weights,T *out,
//          int Tr, int Tc, int Tm, int Tn, int K)
//{
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
//void load(int * buffer_in, int rowt, int colt, int inH1, int inW1) 		//MRAM->WRAM
//{
//    //load input，实际还可进一步减少加载
////    buffer_in.clear();
////    buffer_in.resize();
//    for (int i = 0; i < C_in; ++i)
//    {
//        for (int j = 0; j < inH1; ++j)
//        {
//            mram_read(&input[i * H_ * W_ + (j + rowt) * W_ + colt],
//                      &buffer_in[i * inH1 * inW1 + j * inW1], ALIGN8(inW1 * sizeof(int)));
//        }
//    }
//
//}

//__dma_aligned T buffer_in[C_in * 8 * 8];
//__dma_aligned T buffer_weights[C_out * C_in * K_ * K_];
//__dma_aligned T buffer_out[C_out * out_size * out_size];

//conv2d : 150528 + 3211264 + 1792 = 3363584 ( 3.207763671875 M)--- 0.86704128
//{ 64, 3, CONV_SIZE, CONV_SIZE },
//conv2d_1 : 3211264 + 3211264 + 36928 = 6459456 ( 6.16021728515625 M)--- 18.49688064
//{ 64, 64, CONV_SIZE, CONV_SIZE },
//conv2d_5 : 802816 + 802816 + 590080 = 2195712 ( 2.093994140625 M)--- 18.49688064
//{ 256, 256, CONV_SIZE, CONV_SIZE },
