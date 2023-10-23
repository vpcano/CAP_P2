    #include <iostream>
    #include <stdint.h>
    #include <math.h>
    #include <sys/time.h>
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image_write.h"
    #include "cuda.h"
    #include "cuda_runtime.h"

    #define BLOCK_SIZE_SIDE 16

    __global__ void rgbToGrayscale(uint8_t *rgb_image, uint8_t *grey_image, int width, int height, int nchannels) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int grey_idx = y * width + x;
            int rgb_idx = grey_idx * nchannels;
            unsigned char *offset = rgb_image + rgb_idx;

            grey_image[grey_idx] = (int)(0.2989*offset[0] + 0.5870*offset[1] + 0.1140*offset[2]);
        }
    }

    int main(int argc, char *argv[]) {
        int width, height, nchannels;
        struct timeval fin,ini;

        if (argc < 2) {
            printf("Usage: %s <image1> [<image2> ...]\n", argv[0]);
            return -1;
        }

        // For each image
        for (int file_i = 1; file_i < argc; file_i++) {
            printf("[info] Processing %s\n", argv[file_i]);

            // Reading file
            uint8_t *rgb_image = stbi_load(argv[file_i], &width, &height, &nchannels, 4);
            if (!rgb_image) {
                perror("Image could not be opened");
            }

            // Allocating memory
            uint8_t *grey_image = (uint8_t*) malloc(width * height);
            if (!grey_image) {
                perror("Could not allocate memory");
                stbi_image_free(rgb_image);
            }

            // - Filenames
            for (int i = strlen(argv[file_i]) - 1; i >= 0; i--)
            {
                if (argv[file_i][i] == '.')
                {
                    argv[file_i][i] = 0;
                    break;
                }
            }

            char *grey_image_filename = 0;
            asprintf(&grey_image_filename, "%s_grey.jpg", argv[file_i]);
            if (!grey_image_filename) {
                perror("Could not allocate memory");
                stbi_image_free(rgb_image);
                exit(-1);
            }

            /****** Computations ******/
            printf("[info] %s: width=%d, height=%d, nchannels=%d\n", argv[file_i], width, height, nchannels);

            if (nchannels != 3 && nchannels != 4)
            {
                printf("[error] Num of channels=%d not supported. Only three (RGB), four (RGBA) are supported.\n", nchannels);
                continue;
            }

            // Allocate memory on GPU
            uint8_t *d_rgb_image;
            uint8_t *d_grey_image;
            cudaMalloc((void**)&d_rgb_image, width * height * 4);
            cudaMalloc((void**)&d_grey_image, width * height);

            gettimeofday(&ini,NULL);

            // Copy RGB image to GPU
            cudaMemcpy(d_rgb_image, rgb_image, width * height * 4, cudaMemcpyHostToDevice);

            // Calculate block and grid dimensions
            dim3 blockSize(BLOCK_SIZE_SIDE, BLOCK_SIZE_SIDE);
            dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

            // Launch the kernel
            rgbToGrayscale<<<gridSize, blockSize>>>(d_rgb_image, d_grey_image, width, height, 4);

            // Copy the result back to the host
            cudaMemcpy(grey_image, d_grey_image, width * height, cudaMemcpyDeviceToHost);

            // Save the grayscale image
            stbi_write_jpg(grey_image_filename, width, height, 1, grey_image, 10);

            gettimeofday(&fin,NULL);
            printf("Tiempo: %f\n", ((fin.tv_sec*1000000+fin.tv_usec)-(ini.tv_sec*1000000+ini.tv_usec))*1.0/1000000.0);

            // Free allocated memory
            cudaFree(d_rgb_image);
            cudaFree(d_grey_image);
            stbi_image_free(rgb_image);
            free(grey_image_filename);
            free(grey_image);

        }

        return 0;

    }
