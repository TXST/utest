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


#define TASK_NUM (14)   //14 + 1

#define CIN 64
#define COUT 1

//WRAM���棬��̬����
__dma_aligned T* buffer_in;
__dma_aligned T* buffer_weight;
__dma_aligned T* buffer_out;


//MRAM�о�̬����
#define INPUT_SIZE  (224 * 4 * 14)    //input   ����һ�飨cin=4��14lines��������ļ��ر����㸲��
#define WEIGHT_SIZE (64)   //weight  ÿͨ��1����
#define OUTPUT_SIZE (224 * 14)    //output  1��ͨ��
#define SIZE_224x4  (224 * 4)
#define SIZE_224x16  (224 * 16)

__mram_noinit T weight[WEIGHT_SIZE];
__mram_noinit T input[INPUT_SIZE];
__mram_noinit T output[OUTPUT_SIZE];


T ztb = 0;
BARRIER_INIT(my_barrier, TASK_NUM + 1);

extern void inner_p(T* xx,T* oo,int16_t size,T* BB);



int main() {

    if(me() == 0){
        buffer_in = mem_alloc(INPUT_SIZE);
        buffer_weight = mem_alloc(WEIGHT_SIZE);
        buffer_out = mem_alloc(OUTPUT_SIZE);
        // for(int i = 0;i < 8192;i ++) buffer_out[i] = 1;

        mram_read(weight, buffer_weight, 64);
        //ƫ�Ƽ��㲻��ȷ������Ӱ�����
        for(int i = 0,offset = 0;i < TASK_NUM;i ++,offset += SIZE_224x16){
            mram_read(input + offset, buffer_in + offset, 2048);
            mram_read(input + offset + 2048, buffer_in + offset + 2048, SIZE_224x16 - 2048);
        }
        
    }
    barrier_wait(&my_barrier);

    // printf("thread-%d--stack--%d\n",me(),check_stack());

    int in_offset = me() * SIZE_224x16;
    int out_offset = me() * 224;

    for(int z = 0;z < 224;z += 14){
        for (T i = 0; i < 64; i += 16)
        {
            inner_p(buffer_in + in_offset,buffer_out + out_offset,224,buffer_weight);
        }
        
    }

    
    // if(me() == 0) printf("%d\t",buffer_out[6]);

        // mram_write(buffer_out + write_offset + cout_offset, outputAB + write_offset + cout_offset, (size));


        //ѭ���û���A��B�����ݣ�����
        // int16_t cin_ = 0,wsize_cout = CIN * 9;
        // while(finish != 1) {
        //     //�Ȼ�����������
        //     barrier_wait(&my_barrier);

        //     T myline = me();

        //     //ѭ��ÿ�δ���һ��cin���ϵ�14�У�14�̣߳���
        //     for(T b = 0;b < IN_GROUPS;b ++){
        //         //���̸߳������
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