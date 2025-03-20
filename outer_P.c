#include <stdio.h>
#include <mram.h>
#include <stdbool.h>
#include <stdint.h>
#include <defs.h>

#include <alloc.h>
#include <stddef.h>
#include <stdint.h>
#include <barrier.h>


#define T int8_t


#define TASK_NUM (16)   

#define CIN 64
#define COUT 1

//WRAM缓存，动态分配
__dma_aligned T* buffer_in;
__dma_aligned T* buffer_weight;
__dma_aligned T* buffer_out;


//MRAM中静态分配
#define INPUT_SIZE  (64)    //x
#define WEIGHT_SIZE (128)   //B
#define OUTPUT_SIZE ((8) * 1024)    //h

__mram_noinit T weight[WEIGHT_SIZE];
__mram_noinit T input[INPUT_SIZE];
__mram_noinit T output[OUTPUT_SIZE];


T ztb = 0;
BARRIER_INIT(my_barrier, TASK_NUM);

extern void outer_p(T* xx,T* oo,int16_t size,T* BB);



int main() {

    if(me() == 0){
        buffer_in = mem_alloc(INPUT_SIZE);
        buffer_weight = mem_alloc(WEIGHT_SIZE);
        buffer_out = mem_alloc(OUTPUT_SIZE);
        // for(int i = 0;i < 8192;i ++) buffer_out[i] = 1;

        mram_read(weight, buffer_weight, 128);
        mram_read(input, buffer_in, 64);
    }
    barrier_wait(&my_barrier);

    // printf("thread-%d--stack--%d\n",me(),check_stack());
    //每人8个点，B = 128 = 16 * 8
    //x = 64 * 64(dpu)
    outer_p(buffer_in,buffer_out + me() * 8,64,buffer_weight + me() * 8);
    // if(me() == 0) printf("%d\t",buffer_out[6]);

        // mram_write(buffer_out + write_offset + cout_offset, outputAB + write_offset + cout_offset, (size));




        //循环拿缓冲A或B的数据，算卷积
        // int16_t cin_ = 0,wsize_cout = CIN * 9;
        // while(finish != 1) {
        //     //等缓冲区加载完
        //     barrier_wait(&my_barrier);

        //     T myline = me();

        //     //循环每次处理一个cin（上的14行（14线程））
        //     for(T b = 0;b < IN_GROUPS;b ++){
        //         //本线程负责的行
        //         T* in = buffer_in + (b * TASK_NUM + myline - 1) * OUTHW;

        //         if(myline - 1 == 0 && groups == 0) first = 1;
        //         else first = 0;
        //         if(myline - 1 == 13 && groups == output_groups[conv_num] - 1) last = 1;
        //         else last = 0;


        //         for(T j = 0;j < COUT;j ++){


        //             conv_1d(in,out,OUTHW,wei);

        //         }
        //     }
        // }

    return 0;
}