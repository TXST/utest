
#ifdef _OPENMP
#include <omp.h>
#endif

#include <dpu>
#include <mutex>
#include "omp.h"
#include "iostream"
#include "vector"
#include "string"
#include <random>


extern "C" {
#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
}



using namespace std;


#ifndef DPU_BINARY
#define DPU_BINARY "./task2"
#endif

#define T int8_t

#define DPU_NUM 528

#define IMG_SIZE 224
#define CONV_SIZE 3


static inline double my_clock(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (1.0e-9 * t.tv_nsec + t.tv_sec);
}


//vector<vector<vector<T> > > image(3, vector<vector<T>>(IMG_SIZE, vector<T>(IMG_SIZE, 0)));;

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
//权重w和偏置b
vector<vector<vector<vector<vector<T> > > > > wc;
vector<vector<T> > bc;
int dshape[3][2] = {
        { 25088, 4096 },
        { 4096, 4096 },
        { 4096, 1000 }
};
vector<vector<vector<T> > > wd;
vector<vector<T> > bd;


// Blocks for intermediate convolutions
//int mem_block_shape[3] = {512, IMG_SIZE, IMG_SIZE};
//vector<vector<vector<T> > > mem_block1(512, vector<vector<T>>(IMG_SIZE, vector<T>(IMG_SIZE, 0)));
//vector<T> mem_block2(512 * IMG_SIZE * IMG_SIZE, 0);
// Blocks for dense flatten layers
int mem_block_dense_shape = {512 * 7 * 7};
vector<T> mem_block1_dense(512 * 7 * 7,0);
vector<T> mem_block2_dense(512 * 7 * 7,0);



void reset_mem_block(vector<vector<vector<T> > > &mem_block) {

    int i, j, k;
    for (i = 0; i < mem_block.size(); i++) {
        for (j = 0; j < mem_block[i].size(); j++) {
            for (k = 0; k < mem_block[i][j].size(); k++) {
                mem_block[i][j][k] = 0;
            }
        }
    }
}


void reset_mem_block_dense(vector<T> &mem) {
    int i;
    for (i = 0; i < mem_block_dense_shape; i++) {
        mem[i] = 0;
    }
}


void init_memory() {

    int i, j, k, l;
    // Init convolution weights
    wc.resize(13);
    bc.resize(13);
    for (l = 0; l < 13; l++) {
        wc[l].resize(cshape[l][0]);
        for (i = 0; i < cshape[l][0]; i++) {
            wc[l][i].resize(cshape[l][1]);
            for (j = 0; j < cshape[l][1]; j++) {
                wc[l][i][j].resize(cshape[l][2]);
                for (k = 0; k < cshape[l][2]; k++) {
                    wc[l][i][j][k].resize(cshape[l][3]);
                }
            }
        }
        bc[l].resize(cshape[l][0]);
    }

    // Init dense weights
    wd.resize(3);
    bd.resize(3);
    for (l = 0; l < 3; l++) {
        wd[l].resize(dshape[l][0]);
        for (i = 0; i < dshape[l][0]; i++) {
            wd[l][i].resize(dshape[l][1]);
        }
        bd[l].resize(dshape[l][1]);
    }

}


void free_memory() {

}

void check_output(vector<T>in,vector<T>out,int level,int size){

    for(int j = 0;j < 2;j ++){
        printf("\ninput:\n");
        for(int z = 0;z < 3;z ++){
            for(int t = 0;t < 3;t ++){
                printf("%d ",in[j * size * size + z * size + t]);
            }
        }
        printf("\nweight:\n");
        for(int z = 0;z < 3;z ++){
            for(int t = 0;t < 3;t ++){
                printf("%d ",wc[level][0][j][z][t]);
            }
        }
        printf("\noutput:\n");
        for(int z = 0;z < 3;z ++){
            for(int t = 0;t < 3;t ++){
                printf("%d ",out[j * size * size + z * size + t]);
            }
        }
    }

    printf("\n");
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

//    for(int i = 0;i < 3;i ++) {
//        for(int j = 0;j < 3;j ++)
//            printf("%d %d %d\n",wc[12][0][i][j][0],
//                   wc[12][0][i][j][1],wc[12][0][i][j][2]);
//    }

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


void read_image(char *in_file,vector<vector<vector<T> > > &image) {

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

T zeropad[IMG_SIZE + 2][IMG_SIZE + 2] = {0};

void convolution_3_x_3(vector<vector<T> >matrix, vector<vector<T> >kernel, vector<vector<T> >&out, int size) {
    int i, j;
    T sum;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            zeropad[i + 1][j + 1] = matrix[i][j];
        }
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            sum = zeropad[i][j] * kernel[0][0] +
                  zeropad[i + 1][j] * kernel[1][0] +
                  zeropad[i + 2][j] * kernel[2][0] +
                  zeropad[i][j + 1] * kernel[0][1] +
                  zeropad[i + 1][j + 1] * kernel[1][1] +
                  zeropad[i + 2][j + 1] * kernel[2][1] +
                  zeropad[i][j + 2] * kernel[0][2] +
                  zeropad[i + 1][j + 2] * kernel[1][2] +
                  zeropad[i + 2][j + 2] * kernel[2][2];
            out[i][j] += sum;
        }
    }

}

//
void add_bias_and_relu(vector<vector<T> >&out, T bs, int size) {
    int i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            out[i][j] += bs;
            if (out[i][j] < 0)
                out[i][j] = 0;
        }
    }
}


T max_of_4(T a, T b, T c, T d) {
    if (a >= b && a >= c && a >= d) {
        return a;
    }
    if (b >= c && b >= d) {
        return b;
    }
    if (c >= d) {
        return c;
    }
    return d;
}


void maxpooling(vector<vector<T> >&out, int size) {
    int i, j;
    for (i = 0; i < size; i+=2) {
        for (j = 0; j < size; j+=2) {
            out[i / 2][j / 2] = max_of_4(out[i][j], out[i + 1][j], out[i][j + 1], out[i + 1][j + 1]);
        }
    }
}


void conv1(vector<vector<vector<T> > > image,vector<vector<vector<T> > > &out){

    int i, j;
    int level, cur_size;

    reset_mem_block(out);

    level = 0;
    cur_size = IMG_SIZE;
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(image[j], wc[level][i][j], out[i], cur_size);

        }
        add_bias_and_relu(out[i], bc[level][i], cur_size);
    }

}


int align8(int size){

    if(size % 8 == 0) return size;
    else return size / 8 * 8 + 8;
}


#define STAGE 4
#define PAD(H) (H)

int lvls = 2;      //要计算的卷积层数
mutex mtx[STAGE];
int thread_idx = 0;

struct dpu_set_t set[13];

char *weights_file = "weights1.txt";
char *image_files[] = {"cat1.txt","cat2.txt"};
char *output_file = "results.txt";


//每层卷积使用的dpu数
T split[13] = {1,64,32,64,32,64,64, //第一层不管
               32,64,64,16,16,16};  //共申请528个dpu
//是否需要pooling
T pool[13] = {0,1,0,1,0,0,1,
              0,0,1,0,0,1};
//输出尺寸(=输入)
int out_HW[13] = {224, 224, 112, 112, 56, 56, 56,
                  28, 28, 28, 14, 14, 14};

//13层卷积结果
vector<vector<T > > conv_res(13);


class Worker{

public:
    Worker(int pid): pid(pid) {
        getVGG_predict();
    }
    int64_t pid;    //8字节给dpu

    int getVGG_predict() {

        int label;
        struct dpu_set_t dpu;
        uint32_t idx,idz = 0;
        vector<vector<vector<T> > > image(3, vector<vector<T>>(IMG_SIZE, vector<T>(IMG_SIZE, 0)));;
        vector<vector<vector<T> > > mem_block1(64, vector<vector<T>>(IMG_SIZE, vector<T>(IMG_SIZE, 0)));
        double start,end;

        //--------------------conv1 in cpu------------------------
        //加载img ———— 这玩意如果很慢就单开一个线程
        read_image(image_files[pid],image);
        //为了负载均衡，第一层在cpu算

        //TODO：---------mem_block1--数组或线程内部变量

        conv1(image,mem_block1);

        printf("conv1 done~\n");
        //conv1结果作为输入广播给第一个set
//        int out_size = cshape[0][0] * PAD(out_HW[0]) * PAD(out_HW[0]);
        int out_size = cshape[0][0] * out_HW[0] * PAD(out_HW[0]);   //为了dpu 1D conv方便，这里只padding行方向---不pad了
        conv_res[0].resize(out_size); //64 * 226 * 226
        
        for(int i = 0;i < cshape[0][0];i ++)
            for(int j = 0;j < out_HW[0];j ++)
                for(int k = 0;k < out_HW[0];k ++)
                    conv_res[0][idz ++] = mem_block1[i][j][k];
//                    conv_res[0][i * PAD(out_HW[0]) * PAD(out_HW[0])
//                                + (j + 1) * PAD(out_HW[0]) + (k + 1)] = mem_block1[i][j][k];      //padding
//                    conv_res[0][i * out_HW[0] * PAD(out_HW[0])
//                                + j * PAD(out_HW[0]) + (k + 1)] = mem_block1[i][j][k];

        //--------------------conv2--------------------------

        mtx[0].lock();
        //用来区分AB缓冲区
        thread_idx ++;
        pid = thread_idx;
        cout << "Worker" << pid << "---thread" << omp_get_thread_num() << endl;
        char * inputAB;
        char * outputAB;
        if(pid % 2){
            inputAB = "inputA";
            outputAB = "outputA";
        }else{
            inputAB = "inputB";
            outputAB = "outputB";
        }

        //start transfer
        DPU_ASSERT(dpu_broadcast_to(set[1], inputAB, 0, &conv_res[0][0],
                                    (out_size * sizeof(T)), DPU_XFER_DEFAULT));
        //告诉dpu是用A0还是B1----锁保证线程有序，以后的层可以用相同的缓冲区
        DPU_ASSERT(dpu_broadcast_to(set[1], "pid", 0,
                                    &pid, sizeof(pid), DPU_XFER_DEFAULT));
        //先拿后锁，再放前锁，保证执行顺序
        mtx[1].lock();
        mtx[0].unlock();

       start = my_clock();

        DPU_ASSERT(dpu_launch(set[1], DPU_SYNCHRONOUS));
        //TODO：最后把log去掉
        // DPU_FOREACH(set[1], dpu) {
        //     DPU_ASSERT(dpu_log_read(dpu, stdout));
        // }

        end = my_clock();

        printf(" %.2e secs.\n", end - start);

        printf("conv2 done~\n");

        mtx[2].lock();
        mtx[1].unlock();

        //------------------------conv3------------------------------
        //from set1 to set2
        out_size = cshape[1][0] * out_HW[1] * out_HW[1];
        conv_res[1].resize(out_size);
        //cpu receive from set1 ---- 看下顺序对不对，不对就要加序号，cpu再倒一次内存

        DPU_FOREACH(set[1], dpu, idx) {
//            DPU_ASSERT(dpu_prepare_xfer(dpu, &mem_block2[idx * (out_size / split[1])]));
            DPU_ASSERT(dpu_prepare_xfer(dpu, &conv_res[1][idx * (out_size / split[1])]));
        }
        DPU_ASSERT(dpu_push_xfer(set[1], DPU_XFER_FROM_DPU, outputAB,
                                 0, (out_size / split[1]) * sizeof(T), DPU_XFER_DEFAULT));

        // check_output(conv_res[0],conv_res[1],1,out_HW[1]);

        //padding
//        out_size = cshape[1][0] * out_HW[1] * PAD(out_HW[1]);
//        conv_res[1].resize(out_size);
//        for(int i = 0;i < cshape[1][0];i ++)
//            for(int j = 0;j < out_HW[1];j ++)
//                for(int k = 0;k < out_HW[1];k ++)
//                    conv_res[1][i * out_HW[1] * PAD(out_HW[1])
//                                + j * PAD(out_HW[1]) + (k + 1)]
//                                = mem_block2[i * out_HW[1] * out_HW[1] + j * out_HW[1] + k];

//        //cpu transfer to set2
//        DPU_ASSERT(dpu_broadcast_to(set[2], inputAB, 0, &conv_res[1][0],
//                                    (out_size * sizeof(T)), DPU_XFER_DEFAULT));
//
//
//        mtx[3].lock();
        mtx[2].unlock();



//        DPU_ASSERT(dpu_launch(set[2], DPU_SYNCHRONOUS));
//
//        DPU_FOREACH(set[2], dpu) {
//            DPU_ASSERT(dpu_log_read(dpu, stdout));
//        }
//
//        printf("conv3 done~\n");
////        mtx[4].lock();
//        mtx[3].unlock();

        //------------------------conv4------------------------------


        return label;
    }
};


int main() {


    struct dpu_set_t dpu;
    uint32_t idx;
    double start,end;
    // 1. 创建随机数引擎（用随机设备初始化种子）
    std::random_device rd;  // 硬件随机数生成器（种子）
    std::mt19937 gen(rd()); // Mersenne Twister 引擎

    // 2. 定义分布范围：0 到 127（共 128 个整数）
    std::uniform_int_distribution<int> dist(0, 127);
    T BB[64 * 64 * 4],xx[256 * 128],oo[128 * 64 * 640] = {0};
    int cnt = 0;

    //------------------outer_P----------------------
    //CPU    
    for(int i = 0;i < 128;i ++ ) BB[i] = dist(gen);
    for(int i = 0;i < 256 * 128;i ++ ) xx[i] = dist(gen);

    start = my_clock();

    #pragma omp parallel for num_threads(16)
    for(int i = 0; i < 128; i++) {
        for(int j = 0; j < 256 * 128; j++) {
            int index = i * (256 * 128) + j;  // 线程安全的索引计算
            oo[index] += BB[i] * xx[j];
        }
    }
    // for(int i = 0;i < 128;i ++ ){
    //     for(int j = 0;j < 64 * 640;j ++ ) oo[cnt ++] += BB[i] * xx[j];
    // }
    end = my_clock();
    printf(" %.4e secs.\n", end - start);
    // std::cout << omp_get_max_threads() << std::endl;
    printf(" %.d \n", oo[128 * 64 *64 - 1]);

    //DPU
    for (int zzz = 128; zzz < 2561; zzz+=128)
    {
        DPU_ASSERT(dpu_alloc(zzz, NULL, &set[1])); //申请
        DPU_ASSERT(dpu_load(set[1], DPU_BINARY, NULL));//装程序
        DPU_FOREACH(set[1], dpu, idx) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &xx[0]));
        }
        DPU_ASSERT(dpu_push_xfer(set[1], DPU_XFER_TO_DPU, "input", 0,
                                 (256 * sizeof(T)), DPU_XFER_DEFAULT));
        DPU_FOREACH(set[1], dpu, idx) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &BB[0]));
        }
        DPU_ASSERT(dpu_push_xfer(set[1], DPU_XFER_TO_DPU, "weight", 0,
                                 (128 * sizeof(T)), DPU_XFER_DEFAULT));
        start = my_clock();
    
        DPU_ASSERT(dpu_launch(set[1], DPU_SYNCHRONOUS));
        end = my_clock();
        printf(" %.2e \n", end - start);

        DPU_ASSERT(dpu_free(set[1]));
    }
    

    printf("out_p done~\n");

    //----------------inner_P--------------------
/*
    for(int i = 0;i < 64 * 64 * 4;i ++ ) BB[i] = dist(gen);
    for(int i = 0;i < 224 * 4 * 14;i ++ ) xx[i] = dist(gen);

    //CPU


    //DPU
    DPU_ASSERT(dpu_alloc(64 * 4, NULL, &set[1])); //申请
    DPU_ASSERT(dpu_load(set[1], DPU_BINARY, NULL));//装程序
    DPU_FOREACH(set[1], dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &xx[0]));
    }
    DPU_ASSERT(dpu_push_xfer(set[1], DPU_XFER_TO_DPU, "input", 0,
                             (224 * 4 * 14 * sizeof(T)), DPU_XFER_DEFAULT));
    DPU_FOREACH(set[1], dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &BB[idx * 64]));
    }
    DPU_ASSERT(dpu_push_xfer(set[1], DPU_XFER_TO_DPU, "weight", 0,
                             (64 * sizeof(T)), DPU_XFER_DEFAULT));
    start = my_clock();

    DPU_ASSERT(dpu_launch(set[1], DPU_SYNCHRONOUS));
    //TODO：最后把log去掉
    // DPU_FOREACH(set[1], dpu) {
    //     DPU_ASSERT(dpu_log_read(dpu, stdout));
    // }
    end = my_clock();
    printf(" %.2e secs.\n", end - start);
    printf("inner_P done~\n");

 */

 
    return 0;


    //-------------------------------------------

    init_memory();
//  1.加载weight
    read_weights(weights_file, lvls);

//  2.申请dpu，分配数据,（img和kernel(输出通道/split)）,CPU DRAM数据传到DPU MRAM
//  3.dpu需要把数据搬到WRAM才能计算(可使用fuse,算到最后一层conv)

    T weights[512 * 512 * 3 * 3];

    //从第二层开始，分配12组dpu，和每组上面的weight
    for(T z = 1;z < lvls;z ++){

        DPU_ASSERT(dpu_alloc(split[z], NULL, &set[z])); //申请
        DPU_ASSERT(dpu_load(set[z], DPU_BINARY, NULL));//装程序

        //取出每一层，按计算负载切输出通道，分给不同数量的dpu
        int cout_per_dpu = cshape[z][0] / split[z];
        int size = (cout_per_dpu * cshape[z][1] * cshape[z][2] * cshape[z][3]);
        int idz = 0;
        //之前用申请指针方式拿到的数组不是连续的空间（每次malloc出来的可能不挨着？）
        for(int i = 0;i < cshape[z][0];i ++)
            for(int j = 0;j < cshape[z][1];j ++)
                for(int k = 0;k < cshape[z][2];k ++)
                    for(int t = 0;t < cshape[z][3];t ++)
                        weights[idz ++] = wc[z][i][j][k][t];

        DPU_FOREACH(set[z], dpu, idx) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &weights[idx * size]));
        }
        DPU_ASSERT(dpu_push_xfer(set[z], DPU_XFER_TO_DPU, "weight", 0,
                                 (size * sizeof(T)), DPU_XFER_DEFAULT));
        int64_t conv_num = z;     //告诉dpu要算哪一层
        DPU_ASSERT(dpu_broadcast_to(set[z], "conv_num", 0,
                                    &conv_num, sizeof(conv_num), DPU_XFER_DEFAULT));

    }


//begin pipeline
#pragma omp parallel for

    for(int i = 0;i < 1;i ++){
        Worker w(i);
    }



    //  4.所有结果回传拼在一起，进入fc


    for(T z = 1;z < lvls;z ++)
        DPU_ASSERT(dpu_free(set[z]));

    free_memory();
    return 0;
}



//#pragma omp parallel sections
//    {
//#pragma omp section
//        {
//            for (int i = 0; i < 3; ++i) {
//                printf("hhh, here is thread %d and section 1\n", omp_get_thread_num());
//            }
//        }
//#pragma omp section
//        {
//            for (int i = 0; i < 3; ++i) {
//                printf("hhh, here is thread %d and section 2\n", omp_get_thread_num());
//            }
//        }
//#pragma omp section
//        {
//            for (int i = 0; i < 3; ++i) {
//                printf("hhh, here is thread %d and section 3\n", omp_get_thread_num());
//            }
//        }
//    }


void add_bias_and_relu_flatten(vector<T> &out, vector<T> bs, int size, int relu) {
    int i;
    for (i = 0; i < size; i++) {
        out[i] += bs[i];
        if (relu == 1) {
            if (out[i] < 0)
                out[i] = 0.0;
        }
    }
}

//根据python那边的情况，注意这里的维度，先收集的是通道维度
void flatten(vector<vector<vector<T> > >in, vector<T>&out, int sh0, int sh1, int sh2) {
    int i, j, k, total = 0;
    for (i = 0; i < sh1; i++) {
        for (j = 0; j < sh2; j++) {
            for (k = 0; k < sh0; k++) {
                out[total] = in[k][i][j];
                total += 1;
            }
        }
    }
}

void dense(vector<T> in, vector<vector<T> >weights, vector<T> &out, int sh_in, int sh_out) {
    int i, j;

    for (i = 0; i < sh_out; i++) {
        T sum = 0.0;
        for (j = 0; j < sh_in; j++) {
            sum += in[j] * weights[j][i];
        }
        out[i] = sum;
    }
}

void mlp(vector<vector<vector<T> > > mem_block1) {

    int i, j;
    int level, cur_size;

    reset_mem_block_dense(mem_block1_dense);
    reset_mem_block_dense(mem_block2_dense);

    level = 12;
    cur_size = 7;


    printf("Flatten\n");

    // Layer 19 (Flatten)
    flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);

    // Layer 20 (Dense)
    level = 0;
    dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
    add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
    reset_mem_block_dense(mem_block1_dense);

    // Layer 21 (Dense)
    level = 1;
    dense(mem_block2_dense, wd[level], mem_block1_dense, dshape[level][0], dshape[level][1]);
    add_bias_and_relu_flatten(mem_block1_dense, bd[level], dshape[level][1], 1);
    reset_mem_block_dense(mem_block2_dense);

    // Layer 22 (Dense)
    level = 2;
    dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
    add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);

    printf("\ninput:\n");
    for(int z = 0;z < 49;z ++){
        printf("%d ",mem_block1_dense[z]);
    }
    printf("\nweight:\n");
    for(int z = 0;z < 3;z ++){
        for(int t = 0;t < 3;t ++){
            printf("%d ",wd[level][z][t]);
        }
    }
    printf("\noutput:\n");
    for(int z = 0;z < 49;z ++){
        printf("%d ",mem_block2_dense[z]);
    }
    printf("\n");


    int idx = 0;
    T max = 0.0;
    for(i = 0;i < 1000;i ++){
        if(mem_block2_dense[i] > max) {
            max = mem_block2_dense[i];
            idx = i;
        }
    }
    printf("\n predict class: %d \n",idx);
}
