#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define IN_HEIGHT  224
#define IN_WIDTH   224
#define IN_CH      64
#define OUT_CH     256  // 64*4
#define T int8_t

static inline double my_clock(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (1.0e-9 * t.tv_nsec + t.tv_sec);
}

void pointwise_conv2d(T input[IN_HEIGHT][IN_WIDTH][IN_CH],
                      T output[IN_HEIGHT][IN_WIDTH][OUT_CH],
                      T weights[OUT_CH][IN_CH],
                      T biases[OUT_CH])
{
    #pragma omp parallel for //num_threads(16)
    for (int h = 0; h < IN_HEIGHT; h++) {
        for (int w = 0; w < IN_WIDTH; w++) {
            for (int c_out = 0; c_out < OUT_CH; c_out++) {
                // 初始化累加值为偏置项
                T acc = biases[c_out];
                
                // 1x1卷积核计算
                for (int c_in = 0; c_in < IN_CH; c_in++) {
                    acc += input[h][w][c_in] * weights[c_out][c_in];
                }
                
                // 使用ReLU激活函数
                output[h][w][c_out] = acc > 0.0f ? acc : 0.0f;
            }
        }
    }
}

int main() {
    // 动态分配内存（示例）
    T (*input)[IN_WIDTH][IN_CH] = malloc(sizeof(T[IN_HEIGHT][IN_WIDTH][IN_CH]));
    T (*output)[IN_WIDTH][OUT_CH] = malloc(sizeof(T[IN_HEIGHT][IN_WIDTH][OUT_CH]));
    T (*weights)[IN_CH] = malloc(sizeof(T[OUT_CH][IN_CH]));
    T *biases = malloc(sizeof(T[OUT_CH]));
    double start,end;

    // 这里需要初始化权重和偏置（示例）
    // ...

    // 执行卷积操作
    start = my_clock();

    pointwise_conv2d(input, output, weights, biases);

    end = my_clock();

    printf(" %.2e secs.\n", end - start);

    printf("conv2 done~\n");

    // 释放内存
    free(input);
    free(output);
    free(weights);
    free(biases);

    return 0;
}