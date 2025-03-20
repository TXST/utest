#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define N  1024

__global__ void vector_dot_product(float *a, float *b, float *c) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    c[i] = a[i] * b[i];
  }

}

int main() {

  float a[N], b[N], c[N];
  
    printf("Hello ztb!\n");

  // 初始化 a 和 b
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i + 1;
  }


float *dev_a, *dev_b, *dev_c;

  cudaMalloc(&dev_a, N * sizeof(float));
  cudaMalloc(&dev_b, N * sizeof(float));
  cudaMalloc(&dev_c, N * sizeof(float));

  cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);


  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  vector_dot_product<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c);

  cudaDeviceSynchronize();

  cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);

  // 打印结果
  for (int i = 0; i < N; i++) {
    printf("c[%d] = %f\n", i, c[i]);
  }

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

        cudaDeviceReset();//重置CUDA设备释放程序占用的资源
    system("pause");

  return 0;
}