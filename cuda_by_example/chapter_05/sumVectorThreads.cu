#include "../common/book.h"

#define N 20

__global__ void add(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    if(tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    //allocate memory on GPU
    HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));

    //fill arrays 'a' and 'b' on CPU
    for(int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

    //copy arrays 'a' and 'b' on GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

    add<<<1,N>>>(dev_a, dev_b, dev_c);

    //copy array 'c' back from GPU to CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));

    //display results
    for(int i=0; i<N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    //free mem allocated on gpu
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}