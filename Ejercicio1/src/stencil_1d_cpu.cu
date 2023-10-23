    #include <iostream>
    #include <algorithm>
    #include <sys/time.h>
    using namespace std;

    #define RADIUS 3

    void stencil_1D(int *in, int *out, int N) {
        for (int i=0; i<N; i++) {
            out[i] = 0;
            for (int offset=-RADIUS; offset<=RADIUS; offset++) {
                if (i+offset >= 0 && i+offset <=N) {
                    out[i] += in[i+offset];
                }
            }
        }
    }


    void fill_ints(int *x, int n) {
        fill_n(x, n, 1);
    }

    int main(int argc, char *argv[]) {
        int *in, *out;
        int N, size;
        struct timeval t1, t2;
        double t_total;

        if (argc < 2) {
            printf("Error: you must indicate the length of the array\n");
            return 1;
        }

        N = atoi(argv[1]);
        size = N * sizeof(int);

        in = (int*) malloc(size);
        out = (int*) malloc(size);
        fill_ints(in, N);
        fill_ints(out, N);

        gettimeofday(&t1, NULL);
        stencil_1D(in, out, N);
        gettimeofday(&t2, NULL);

        /*
        printf("Output: \n");
        for (int i=0; i<N; i++) {
            printf("%d ", out[i]);
        }
        printf("\n");
        */

        t_total = (t2.tv_sec - t1.tv_sec)*1000000.0 + (t2.tv_usec - t1.tv_usec);
        printf("%d\t%f\n", N, t_total);

        free(in);
        free(out);
        return 0;
    }
