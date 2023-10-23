    #include <iostream>
    #include <algorithm>
    #include <sys/time.h>
    #include "cuda.h"
    #include "cuda_runtime.h"
    using namespace std;

    #define RADIUS 3
    #define BLOCK_SIZE 16

    __global__ void stencil_1D(int *in, int *out, int N) {
        __shared__ int temp[BLOCK_SIZE + 2*RADIUS];
        int gindex = threadIdx.x + blockIdx.x*blockDim.x;
        int lindex = threadIdx.x + RADIUS;

        if (threadIdx.x < RADIUS) {
            if (gindex < RADIUS) {
                temp[lindex - RADIUS] = 0;
            }
            else {
                temp[lindex - RADIUS] = in[gindex - RADIUS];
            }
            if (gindex + BLOCK_SIZE < N) {
                temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
            }
            else {
                temp[lindex + BLOCK_SIZE] = 0;
            }
        }

        if (gindex < N) {
            temp[lindex] = in[gindex];
        }

        __syncthreads();

        if (gindex < N) {
            int result = 0;
            for (int offset=-RADIUS; offset<=RADIUS; offset++) {
                result += temp[lindex + offset];
            }

            out[gindex] = result;
        }
    }


    void fill_ints(int *x, int n) {
        fill_n(x, n, 1);
    }

    int main(int argc, char *argv[]) {
        int *h_in, *h_out;
        int *d_in, *d_out;
        int N, size;
        struct timeval t1, t2;
        double t_total;

        if (argc < 2) {
            printf("Error: you must indicate the length of the array\n");
            return 1;
        }

        N = atoi(argv[1]);
        size = N * sizeof(int);

        h_in = (int*) malloc(size);
        h_out = (int*) malloc(size);
        fill_ints(h_in, N);
        fill_ints(h_out, N);

        cudaMalloc((void**) &d_in, size);
        cudaMalloc((void**) &d_out, size);

        gettimeofday(&t1, NULL);
        cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_out, h_out, size, cudaMemcpyHostToDevice);

        stencil_1D<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_in, d_out, N);

        cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

        gettimeofday(&t2, NULL);

        printf("Output: \n");
        for (int i=0; i<N; i++) {
            printf("%d ", h_out[i]);
        }
        printf("\n");

        t_total = (t2.tv_sec - t1.tv_sec)*1000000.0 + (t2.tv_usec - t1.tv_usec);
        printf("%d\t%f\n", N, t_total);

        free(h_in);
        free(h_out);
        cudaFree(d_in);
        cudaFree(d_out);
        return 0;
    }
