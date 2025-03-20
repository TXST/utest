
#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>

#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <stdint.h>

//#define __USE_POSIX199309
#include <time.h>



#ifndef DPU_BINARY
#define DPU_BINARY "t"
#endif

#define T int8_t
#define DPU_NUM 64

#define IMG_SIZE 224
#define CONV_SIZE 3


T ***image;
int cshape[13][4] = {
        { 64, 3, CONV_SIZE, CONV_SIZE },
        { 64, 64, CONV_SIZE, CONV_SIZE },
        { 128, 64, CONV_SIZE, CONV_SIZE },
        { 128, 128, CONV_SIZE, CONV_SIZE },
        { 256, 128, CONV_SIZE, CONV_SIZE },
        { 256, 256, CONV_SIZE, CONV_SIZE },
        { 256, 256, CONV_SIZE, CONV_SIZE },
        { 512, 256, CONV_SIZE, CONV_SIZE },
        { 512, 512, CONV_SIZE, CONV_SIZE },
        { 512, 512, CONV_SIZE, CONV_SIZE },
        { 512, 512, CONV_SIZE, CONV_SIZE },
        { 512, 512, CONV_SIZE, CONV_SIZE },
        { 512, 512, CONV_SIZE, CONV_SIZE }
};
T *****wc;
T **bc;
int dshape[3][2] = {
        { 25088, 4096 },
        { 4096, 4096 },
        { 4096, 1000 }
};
T ***wd;
T **bd;


// Blocks for intermediate convolutions
int mem_block_shape[3] = {512, IMG_SIZE, IMG_SIZE};
T ***mem_block1;
T ***mem_block2;
// Blocks for dense flatten layers
int mem_block_dense_shape = { 512 * 7 * 7 };
T *mem_block1_dense;
T *mem_block2_dense;


//static inline double my_clock(void) {
//    struct timespec t;
//    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
//    return (1.0e-9 * t.tv_nsec + t.tv_sec);
//}


int align8(int size){

    if(size % 8 == 0) return size;
    else return size / 8 * 8 + 8;
}


void reset_mem_block(T ***mem) {
    int i, j, k;
    for (i = 0; i < mem_block_shape[0]; i++) {
        for (j = 0; j < mem_block_shape[1]; j++) {
            for (k = 0; k < mem_block_shape[2]; k++) {
                mem[i][j][k] = 0.0;
            }
        }
    }
}


void reset_mem_block_dense(T *mem) {
    int i;
    for (i = 0; i < mem_block_dense_shape; i++) {
        mem[i] = 0.0;
    }
}


void init_memory() {
    int i, j, k, l;

    // Init image memory
    image = malloc(3 * sizeof(T**));
    for (i = 0; i < 3; i++) {
        image[i] = malloc(IMG_SIZE * sizeof(T*));
        for (j = 0; j < IMG_SIZE; j++) {
            image[i][j] = malloc(IMG_SIZE * sizeof(T));
        }
    }

    // Init convolution weights
    wc = malloc(13 * sizeof(T****));
    bc = malloc(13 * sizeof(T*));
    for (l = 0; l < 13; l++) {
        wc[l] = malloc(cshape[l][0] * sizeof(T***));
        for (i = 0; i < cshape[l][0]; i++) {
            wc[l][i] = malloc(cshape[l][1] * sizeof(T**));
            for (j = 0; j < cshape[l][1]; j++) {
                wc[l][i][j] = malloc(cshape[l][2] * sizeof(T*));
                for (k = 0; k < cshape[l][2]; k++) {
                    wc[l][i][j][k] = malloc(cshape[l][3] * sizeof(T));
                }
            }
        }
        bc[l] = malloc(cshape[l][0] * sizeof(T));
    }

    // Init dense weights
    wd = malloc(3 * sizeof(T**));
    bd = malloc(3 * sizeof(T*));
    for (l = 0; l < 3; l++) {
        wd[l] = malloc(dshape[l][0] * sizeof(T*));
        for (i = 0; i < dshape[l][0]; i++) {
            wd[l][i] = malloc(dshape[l][1] * sizeof(T));
        }
        bd[l] = malloc(dshape[l][1] * sizeof(T));
    }

    // Init mem_blocks
    mem_block1 = malloc(mem_block_shape[0] * sizeof(T**));
    mem_block2 = malloc(mem_block_shape[0] * sizeof(T**));
    for (i = 0; i < mem_block_shape[0]; i++) {
        mem_block1[i] = malloc(mem_block_shape[1] * sizeof(T*));
        mem_block2[i] = malloc(mem_block_shape[1] * sizeof(T*));
        for (j = 0; j < mem_block_shape[1]; j++) {
            mem_block1[i][j] = malloc(mem_block_shape[2] * sizeof(T));
            mem_block2[i][j] = malloc(mem_block_shape[2] * sizeof(T));
        }
    }
    reset_mem_block(mem_block1);
    reset_mem_block(mem_block2);

    // Init mem blocks dense
    mem_block1_dense = calloc(mem_block_dense_shape, sizeof(T));
    mem_block2_dense = calloc(mem_block_dense_shape, sizeof(T));
}


void free_memory() {
    int i, j, k, l;

    // Free image memory
    for (i = 0; i < 3; i++) {
        for (j = 0; j < IMG_SIZE; j++) {
            free(image[i][j]);
        }
        free(image[i]);
    }
    free(image);

    // Free convolution weights
    for (l = 0; l < 13; l++) {
        for (i = 0; i < cshape[l][0]; i++) {
            for (j = 0; j < cshape[l][1]; j++) {
                for (k = 0; k < cshape[l][2]; k++) {
                    free(wc[l][i][j][k]);
                }
                free(wc[l][i][j]);
            }
            free(wc[l][i]);
        }
        free(wc[l]);
        free(bc[l]);
    }
    free(wc);
    free(bc);

    // Free dense weights
    for (l = 0; l < 3; l++) {
        for (i = 0; i < dshape[l][0]; i++) {
            free(wd[l][i]);
        }
        free(wd[l]);
        free(bd[l]);
    }
    free(wd);
    free(bd);

    // Free memblocks
    for (i = 0; i < mem_block_shape[0]; i++) {
        for (j = 0; j < mem_block_shape[1]; j++) {
            free(mem_block1[i][j]);
            free(mem_block2[i][j]);
        }
        free(mem_block1[i]);
        free(mem_block2[i]);
    }
    free(mem_block1);
    free(mem_block2);

    free(mem_block1_dense);
    free(mem_block2_dense);
}


void read_weights(char *in_file, int lvls) {

    int dval;
    int i, j, k, l, z;
    FILE *iin;
    int total_lvls_read = 0;

    iin = fopen(in_file, "r");
    if (iin == NULL) {
        printf("File %s absent\n", in_file);
        exit(1);
    }

    // Reading convolution weights (store them flipped from begining)
    for (z = 0; z < 13; z++) {

        if (total_lvls_read >= lvls && lvls != -1)
            break;

        printf("Read conv block %d weights\n", z);
        for (i = 0; i < cshape[z][0]; i++) {
            for (j = 0; j < cshape[z][1]; j++) {
                for (k = 0; k < cshape[z][2]; k++) {
                    for (l = 0; l < cshape[z][3]; l++) {
                        fscanf(iin, "%d", &dval);
                        wc[z][i][j][k][l] = dval;
                    }
                }
            }
        }
        for (i = 0; i < cshape[z][0]; i++) {
            fscanf(iin, "%d", &dval);
            bc[z][i] = dval;
        }

        total_lvls_read += 1;
    }

    for(int i = 0;i < 3;i ++) {
        for(int j = 0;j < 3;j ++)
            printf("%d %d %d\n",wc[12][0][i][j][0],
                   wc[12][0][i][j][1],wc[12][0][i][j][2]);
    }

//    printf("%d %d %d\n",wc[0][0][0][0][0],wc[0][0][0][0][1],wc[0][0][0][0][2]);

    // Reading dense weights
    for (z = 0; z < 3; z++) {

        if (total_lvls_read >= lvls && lvls != -1)
            break;

        printf("Read dense block %d weights\n", z);
        for (i = 0; i < dshape[z][0]; i++) {
            for (j = 0; j < dshape[z][1]; j++) {
                fscanf(iin, "%d", &dval);
                wd[z][i][j] = dval;
            }
        }
        for (i = 0; i < dshape[z][1]; i++) {
            fscanf(iin, "%d", &dval);
            bd[z][i] = dval;
        }

        total_lvls_read += 1;
    }

    fclose(iin);
}


void read_image(char *in_file) {
    int i, j, l;
    FILE *iin;
    float dval;

    iin = fopen(in_file, "r");
    if (iin == NULL) {
        printf("File %s absent\n", in_file);

        exit(1);
    }
    printf("reading... %s\n", in_file);

    /* Reading image */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < IMG_SIZE; j++) {
            for (l = 0; l < IMG_SIZE; l++) {
                fscanf(iin, "%f", &dval);
                image[i][j][l] = (T)dval;
            }
        }
    }
//    printf("%d %d %d\n",image[0][0][0],image[1][0][0],image[2][0][0]);

    fclose(iin);
}




//Test:
//通信：
//数据块大小&带宽 ———— 有启动成本，多次传输时，第一次启动成本最高；
//多次测的带宽不一致，差距挺大（跟CPU负载或者DMA通道有关？）
//实测 DRAM ---> MRAM 带宽250MB/s，但是反过来的带宽只有104MB/s

//void trans_block(){
//
//    int size = 8;
//    for(int i = 0;i < 24;i ++){
//
//        double start = my_clock();
//
//        DPU_FOREACH(set, dpu, idx) {
//            DPU_ASSERT(dpu_prepare_xfer(dpu, input));
//        }
//        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "input", 0,
//                                 (size * sizeof(T)), DPU_XFER_DEFAULT));
//
//        double end = my_clock();
//
//        printf("%d\t: %.2e secs.\n", size,end - start);
//
//        size *= 2;
//    }
//}
//行列序 ———— 目前用的一维数组
//并行 ———— 固定传64MB，改变DPU数  ———— 不等长不考虑了，这个形式就是等长的
//在某些数量节点时，耗时会突变（8、64、256、(384、448)、512），没想出原因

//void parallel(){
//
//    for(int i = 1;i < 2560;i *= 2){
//
//        DPU_ASSERT(dpu_alloc(i, NULL, &set));
//
//        DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
//
//        double start = my_clock();
//
//        DPU_FOREACH(set, dpu, idx) {
//            DPU_ASSERT(dpu_prepare_xfer(dpu, input));
//        }
//        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "input", 0,
//                                 (SIZE * sizeof(T)), DPU_XFER_DEFAULT));
//
//        double end = my_clock();
//
//        printf("%d\t: %.2e secs.\n", i,end - start);
//
//        DPU_ASSERT(dpu_free(set));
//
//    }
//}

//广播&for
//All-reduce
//        DPU——CPU——DPU
//        CPU——DPU
//        DPU——CPU
// WRAM——MRAM —————— 完全找不到规律
//void mram_wram(){
//
//    for(int i = 8;i < 60000; i *= 2){
//
//        perfcounter_config(COUNT_CYCLES, true);
//
//        mram_read(input, buffer_in, i * sizeof(T));
//
//        nb_cycles = perfcounter_get();
//
//        printf("%d ---- %d\n",i,nb_cycles);
//
//    }
//}
//        WRAM——RAM
//        CPU取RAM时间 & DPU
//        双缓冲overlap
//计算：
//切分计算负载
//        每层卷积计算量和时间、切通道
//        Fuse
//两层计算 vs all-reduce




int main(int argc, char *argv[]) {

    struct dpu_set_t set[2], dpu;
    int nr_dpus;
    int idx;

    char *weights_file = "weights1.txt";
    char *image_file = "cat1.txt";
    char *output_file = "results.txt";
    int lvls = 13;


//  1.加载img和weight

    //CPU RAM
    init_memory();

//    read_weights(weights_file, lvls);

    read_image(image_file);
    //之前用申请指针方式拿到的数组不是连续的空间（每次malloc出来的可能不挨着？）
    //目前这种分法，重复传输了image
    T input[3 * IMG_SIZE * IMG_SIZE];
    int idz = 0;
    for(int i = 0;i < 3;i ++)
        for(int j = 0;j < IMG_SIZE;j ++)
            for(int k = 0;k < IMG_SIZE;k ++)
                input[idz ++] = image[i][j][k];

    printf("data load complete\n");
    for(int i = 0;i < IMG_SIZE;i ++) printf("%d\t",input[i]);

//  2.申请dpu(64)，分配数据（img和kernel(输出通道/64)）
//  3.dpu需要把数据搬到WRAM,(计算可使用fuse)，算到最后一层conv
//  4.所有结果回传拼在一起，进入fc

for(int i = 0;i < 1;i ++) {

    DPU_ASSERT(dpu_alloc(DPU_NUM, NULL, &set[i])); //申请
    DPU_ASSERT(dpu_load(set[i], DPU_BINARY, NULL));//装程序
    DPU_ASSERT(dpu_get_nr_dpus(set[i], &nr_dpus)); //dpu数
    if (nr_dpus != DPU_NUM) {
        printf("dpu_alloc fail~ bye~");
        return 0;
    }

    // retrieve number of cycles on DPU
//    uint32_t nb_cycles;
//    DPU_FOREACH(set, dpu) {
//        DPU_ASSERT(
//                dpu_copy_from(dpu, "nb_cycles", 0, &nb_cycles, sizeof(uint32_t)));
//    }
//
//    // retrieve DPU frequency
//    uint32_t clocks_per_sec;
//    DPU_FOREACH(set, dpu) {
//        DPU_ASSERT(dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clocks_per_sec,
//                                 sizeof(uint32_t)));
//    }
//
//    printf("DPU cycles: %u\n", nb_cycles);
//    printf("DPU time: %.2e secs.\n", (double)nb_cycles / clocks_per_sec);

//    double start = my_clock();


    DPU_FOREACH(set[i], dpu, idx)
    {
        DPU_ASSERT(dpu_prepare_xfer(dpu, input));
    }
    DPU_ASSERT(dpu_push_xfer(set[i], DPU_XFER_TO_DPU, "input", 0,
                             (IMG_SIZE * IMG_SIZE * 3 * sizeof(T)), DPU_XFER_DEFAULT));

//
//    double end = my_clock();
//
//    printf("Host elapsed time: %.2e secs.\n", end - start);


    T weights[512 * 512 * 3 * 3];

//    int weight_offset = 0;
//    for(int z = 0;z < lvls;z ++){
//        //取出每一层，然后按输出通道平分给dpu
//        //每层维度不同，这里是分别传输；如果只传一次就要拼成一维数组
//        //那边如果用一维数组算比较麻烦(还得定位每一层)，如果还原回多维，就要浪费额外的时间和空间
//        //如果dpu多了，通道少的层会不满足8字节的整数倍，没办法直接传给多维数组，所以每层都用一维数组接，可以align8
//        int cout_per_dpu = cshape[z][0] / DPU_NUM;
//        int size = (cout_per_dpu * cshape[z][1] * cshape[z][2] * cshape[z][3]);
//        idz = 0;
//        for(int i = 0;i < cshape[z][0];i ++)
//            for(int j = 0;j < cshape[z][1];j ++)
//                for(int k = 0;k < cshape[z][2];k ++)
//                    for(int t = 0;t < cshape[z][3];t ++)
//                        weights[idz ++] = wc[z][i][j][k][t];
//
//        DPU_FOREACH(set, dpu, idx) {
//            // 传输可以并行吗？
//            // TODO：看看数据排布会不会影响传输速度
//            DPU_ASSERT(dpu_prepare_xfer(dpu, &weights[idx * size]));
//        }
//
//        char weight_lv[8];
//        sprintf(weight_lv,"%s%d","weight",z + 1);
//        printf("%s\n",weight_lv);
//        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, weight_lv, 0,
//                                 align8(size * sizeof(T)), DPU_XFER_DEFAULT));
//
////        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "weights", weight_offset,
////                                 size * sizeof(T), DPU_XFER_DEFAULT));
////        weight_offset += size;
//
//    }

    DPU_ASSERT(dpu_launch(set[i], DPU_SYNCHRONOUS));

    DPU_FOREACH(set[i], dpu)
    {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }

}

    DPU_ASSERT(dpu_free(set[0]));
    DPU_ASSERT(dpu_free(set[1]));


    free_memory();
    return 0;
}