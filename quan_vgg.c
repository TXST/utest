
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
//#define _OPENMP 1

#define IMG_SIZE 224
#define CONV_SIZE 3
int numthreads;

#define T int8_t


static inline double my_clock(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (1.0e-9 * t.tv_nsec + t.tv_sec);
}


// Weights and image block START
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

// Weights and image block END


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

    // for(int i = 0;i < 3;i ++) {
    //     for(int j = 0;j < 3;j ++)
    //         printf("%d %d %d\n",wc[0][0][i][j][0],
    //                wc[0][0][i][j][1],wc[0][0][i][j][2]);
    // }

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

    /* Reading image */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < IMG_SIZE; j++) {
            for (l = 0; l < IMG_SIZE; l++) {
                fscanf(iin, "%f", &dval);
                image[i][j][l] = (T)dval;
            }
        }
    }
//    for (i = 0; i < 3; i++) printf("\n%d \n",image[i][0][0]);

    fclose(iin);
}


void convolution_3_x_3(T **matrix, T **kernel, T **out, int size) {
    int i, j;
    T sum;
    T zeropad[IMG_SIZE + 2][IMG_SIZE + 2] = {0.0 };

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
void add_bias_and_relu(T **out, T bs, int size) {
    int i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            out[i][j] += bs;
            if (out[i][j] < 0)
                out[i][j] = 0.0;
            // printf("%.12lf\n", out_HW[i][j]);
        }
    }
}


void add_bias_and_relu_flatten(T *out, T *bs, int size, int relu) {
    int i;
    for (i = 0; i < size; i++) {
        out[i] += bs[i];
        if (relu == 1) {
            if (out[i] < 0)
                out[i] = 0.0;
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


void maxpooling(T **out, int size) {
    int i, j;
    for (i = 0; i < size; i+=2) {
        for (j = 0; j < size; j+=2) {
            out[i / 2][j / 2] = max_of_4(out[i][j], out[i + 1][j], out[i][j + 1], out[i + 1][j + 1]);
        }
    }
}

//根据python那边的情况，注意这里的维度，先收集的�?通道维度
void flatten(T ***in, T *out, int sh0, int sh1, int sh2) {
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


void dense(T *in, T **weights, T *out, int sh_in, int sh_out) {
    int i, j;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < sh_out; i++) {
        T sum = 0.0;
        for (j = 0; j < sh_in; j++) {
            sum += in[j] * weights[j][i];
        }
        out[i] = sum;
    }
}


void check_output(T ***in,T ***out,int level){

    for(int j = 0;j < 2;j ++){
        printf("\ninput:\n");
        for(int z = 0;z < 3;z ++){
            for(int t = 0;t < 3;t ++){
                printf("%d ",in[j][z][t]);
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
                printf("%d ",out[j][z][t]);
            }
        }
    }


    printf("\n");
}

double start,end;

void get_VGG16_predict(int lv) {
    int i, j;
    int level, cur_size;

    // Init intermediate memory
    reset_mem_block(mem_block1);
    reset_mem_block(mem_block2);
    reset_mem_block_dense(mem_block1_dense);
    reset_mem_block_dense(mem_block2_dense);

    // Layer 1 (Convolution 3 -> 64)
    level = 0;
    cur_size = IMG_SIZE;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(image[j], wc[level][i][j], mem_block1[i], cur_size);

        }
//        printf("\nbias: %d ",bc[level][i]);
        add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
    }

    if(level + 1 == lv){
        check_output(image,mem_block1,level);
        return;
    }

    // Layer 2 (Convolution 64 -> 64)
    level = 1;

    start = my_clock();

#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size);
        }
//        add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
    }

    end = my_clock();

    printf(" %.2e secs.\n", end - start);

    if(level + 1 == lv){
        check_output(mem_block1,mem_block2,level);
        return;
    }

    reset_mem_block(mem_block1);


    // Layer 3 (MaxPooling)
#pragma omp parallel for schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        maxpooling(mem_block2[i], cur_size);
    }
    cur_size /= 2;

//    check_output(mem_block2,level);

    // Layer 4 (Convolution 64 -> 128)
    level = 2;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size);
        }
        add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
    }

    reset_mem_block(mem_block2);

    // Layer 5 (Convolution 128 -> 128)
    level = 3;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size);
        }
        add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
    }


    reset_mem_block(mem_block1);

    // Layer 6 (MaxPooling)
#pragma omp parallel for schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        maxpooling(mem_block2[i], cur_size);
    }
    cur_size /= 2;



    // Layer 7 (Convolution 128 -> 256)
    level = 4;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size);
        }
        add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
    }
    reset_mem_block(mem_block2);

    // Layer 8 (Convolution 256 -> 256)
    level = 5;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size);
        }
        add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
    }
    reset_mem_block(mem_block1);

    // Layer 9 (Convolution 256 -> 256)
    level = 6;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size);
        }
        add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
    }



    reset_mem_block(mem_block2);

    // Layer 10 (MaxPooling)
#pragma omp parallel for schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        maxpooling(mem_block1[i], cur_size);
    }
    cur_size /= 2;

    // Layer 11 (Convolution 256 -> 512)
    level = 7;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size);
        }
        add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
    }
    reset_mem_block(mem_block1);

    // Layer 12 (Convolution 512 -> 512)
    level = 8;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size);
        }
        add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
    }
    reset_mem_block(mem_block2);

    // Layer 13 (Convolution 512 -> 512)
    level = 9;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size);
        }
        add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
    }
    reset_mem_block(mem_block1);

    // Layer 14 (MaxPooling)
#pragma omp parallel for schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        maxpooling(mem_block2[i], cur_size);
    }
    cur_size /= 2;

    // Layer 15 (Convolution 512 -> 512)
    level = 10;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size);
        }
        add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
    }
    reset_mem_block(mem_block2);

    // Layer 16 (Convolution 512 -> 512)
    level = 11;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block1[j], wc[level][i][j], mem_block2[i], cur_size);
        }
        add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
    }
    reset_mem_block(mem_block1);

    // Layer 17 (Convolution 512 -> 512)
    level = 12;
#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        for (j = 0; j < cshape[level][1]; j++) {
            convolution_3_x_3(mem_block2[j], wc[level][i][j], mem_block1[i], cur_size);
        }
        add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
    }


//    reset_mem_block(mem_block2);

    // Layer 18 (MaxPooling)
#pragma omp parallel for schedule(dynamic,1) num_threads(numthreads)
    for (i = 0; i < cshape[level][0]; i++) {
        maxpooling(mem_block1[i], cur_size);
    }
    cur_size /= 2;

    check_output(mem_block2,mem_block1,level);

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
    for(int z = 0;z < 160;z ++){
        printf("%d ",mem_block2_dense[z]);
    }

    printf("\n");

//    softmax(mem_block2_dense, dshape[level][1]);
    // dump_memory_structure_dense_to_file(mem_block2_dense, dshape[level][1]);

    int idx = 0;
    T max = 0;
    for(i = 0;i < 1000;i ++){
        if(mem_block2_dense[i] > max) {
            max = mem_block2_dense[i];
            idx = i;
        }
    }
    printf("\n predict class: %d \n",idx);
}


void output_predictions(FILE *out, int only_convolution) {
    int i;
    if (only_convolution == 1) {
        for (i = 0; i < 512*7*7; i++) {
            fprintf(out, "%g ", mem_block1_dense[i]);
        }
    }
    else {
        for (i = 0; i < dshape[2][1]; i++) {
            fprintf(out, "%g ", mem_block2_dense[i]);
        }
    }
    fprintf(out, "\n");
}


char *trimwhitespace(char *str)
{
    char *end;

    // Trim leading space
    while (isspace((unsigned char)*str)) str++;

    if (*str == 0)  // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;

    // Write new null terminator
    *(end + 1) = 0;

    return str;
}


int main(int argc, char *argv[]) {
    FILE *file_list, *results;
    char buf[1024];
#ifndef _WIN32
    struct timeval timeStart, timeEnd;
#else
    time_t timeStart, timeEnd;
#endif
    double deltaTime;
    char *weights_file;
    char *image_list_file;
    char *output_file;
    int lvls = -1;


#ifdef _OPENMP
    numthreads = omp_get_num_procs() - 1;
#endif
    if (numthreads < 1)
        numthreads = 1;
    // numthreads = 2;
    printf("Using %d threads\n", numthreads);

//    if (argc != 4 && argc != 5) {
//        printf("Usage: <program.exe> <weights file> <images list file> <output file> <level [optional]>\n");
//        return 0;
//    }
//    weights_file = argv[1];
//    image_list_file = argv[2];
//    output_file = argv[3];
//    if (argc == 5) {
//        lvls = atoi(argv[4]);
//
//    }

    weights_file = "weights1.txt";
    image_list_file = "filelist.txt";
    output_file = "results.txt";
    lvls = 2;

    init_memory();
    file_list = fopen(image_list_file, "r");
    if (file_list == NULL) {
        printf("Check file list location: %s", image_list_file);
        return 1;
    }
    results = fopen(output_file, "w");
    if (results == NULL) {
        printf("Couldn't open file for writing: %s", output_file);
        return 1;
    }

    read_weights(weights_file, lvls);

    while (!feof(file_list)) {

        fgets(buf, 1024, file_list);
        if (strlen(buf) == 0) {
            break;
        }
        // printf("%d\n", strlen(buf));
        read_image(trimwhitespace(buf));
//        normalize_image();
        // dump_image();
        printf("get_VGG16_predict\n");
        get_VGG16_predict(lvls);
//        output_predictions(results);

    }

    free_memory();
    fclose(file_list);
    return 0;
}
