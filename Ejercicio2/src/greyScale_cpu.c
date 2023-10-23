    #include <stdio.h>
    #include <stdint.h>
    #include <math.h>
    #include <sys/time.h>
    #include <x86intrin.h>
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image_write.h"



    int main(int nargs, char **argv)
    {
        int width, height, nchannels;
        struct timeval fin,ini;
        const __m256 coefficients = {0.2989, 0.5870, 0.1140, 0.0, 0.2989, 0.5870, 0.1140, 0.0};

        if (nargs < 2)
        {
            printf("Usage: %s <image1> [<image2> ...]\n", argv[0]);
        }
        // For each image
        // Bucle 0
        for (int file_i = 1; file_i < nargs; file_i++)
        {
            printf("[info] Processing %s\n", argv[file_i]);
            /****** Reading file ******/
            uint8_t *rgb_image = stbi_load(argv[file_i], &width, &height, &nchannels, 4);
            if (!rgb_image)
            {
                perror("Image could not be opened");
            }

            /****** Allocating memory ******/
            // - RGB2Grey
            uint8_t *grey_image = malloc(width * height);
            if (!grey_image)
            {
                perror("Could not allocate memory");
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
            if (!grey_image_filename)
            {
                perror("Could not allocate memory");
                exit(-1);
            }

            /****** Computations ******/
            printf("[info] %s: width=%d, height=%d, nchannels=%d\n", argv[file_i], width, height, nchannels);

            if (nchannels != 3 && nchannels != 4)
            {
                printf("[error] Num of channels=%d not supported. Only three (RGB), four (RGBA) are supported.\n", nchannels);
                continue;
            }

            gettimeofday(&ini,NULL);
            // RGB to grey scale
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i+=4)
                {
                    // Obtain first two pixels (64 bits) and store them as four 8-bit packed
                    // integers each in the lowest half of a 128 bit register
                    __m128i lo_two_pixels = _mm_loadl_epi64((__m128i*)(rgb_image + (i + j*width) * 4));
                    // Do the same with the next two pixels
                    __m128i hi_two_pixels = _mm_loadl_epi64((__m128i*)(rgb_image + (i + 2 + j*width) * 4));

                    // Extend each 8-bit packed integer from each pixel to a 32-bit packed
                    // integer and store each pair of pixels in a 256 bit register
                    __m256i extended_lo_two_pixels = _mm256_cvtepu8_epi32(lo_two_pixels);
                    __m256i extended_hi_two_pixels = _mm256_cvtepu8_epi32(hi_two_pixels);

                    // Convert each 32-bit packed integer to a 32-bit packed float
                    __m256 float_lo_two_pixels = _mm256_cvtepi32_ps(extended_lo_two_pixels);
                    __m256 float_hi_two_pixels = _mm256_cvtepi32_ps(extended_hi_two_pixels);

                    // Multiply each 32-bit packed float with the coefficients (defined
                    // at the begining)
                    __m256 prod_lo_two_pixels = _mm256_mul_ps(float_lo_two_pixels, coefficients);
                    __m256 prod_hi_two_pixels = _mm256_mul_ps(float_hi_two_pixels, coefficients);

                    // Horizontally add the registers in order to achieve the sum of the
                    // products for each pixel
                    __m256 sum = _mm256_hadd_ps(prod_lo_two_pixels, prod_hi_two_pixels);
                    sum = _mm256_hadd_ps(sum, sum);

                    // Reorder the sum results to keep the original pixel order
                    __m256 reordered_sum = _mm256_permutevar8x32_ps(sum, _mm256_set_epi32(7, 6, 3, 2, 5, 1, 4, 0));

                    // Take the four first 32-bit packed floats and convert them to
                    // 32-bit packed integers
                    __m128i grey_pixels = _mm_cvtps_epi32(_mm256_extractf128_ps(reordered_sum, 0));

                    // The following code is valid but requires too much instructions
                    // grey_image[i + j*width] = (uint8_t) _mm_extract_epi8(grey_pixels, 0);
                    // grey_image[i + j*width + 1] = (uint8_t) _mm_extract_epi8(grey_pixels, 4);
                    // grey_image[i + j*width + 2] = (uint8_t) _mm_extract_epi8(grey_pixels, 8);
                    // grey_image[i + j*width + 3] = (uint8_t) _mm_extract_epi8(grey_pixels, 12);

                    // This aproach results in less instructions: each 32-bit integer
                    // is packed into a 16-bit integer and stored next to each other
                    // at the begining of the register. Then the same is done but from
                    // a 16-bit integer to a 8-bit integer, so the result has the four
                    // pixels stored on the first 32 bit of the register. Finally, only
                    // one assignment needs to be done. The grey_image array is casted
                    // to a uint32_t pointer and its value at the corresponding index
                    // is assigned to the first four pixels
                    grey_pixels = _mm_packus_epi32(grey_pixels, grey_pixels);
                    grey_pixels = _mm_packus_epi16(grey_pixels, grey_pixels);
                    *((uint32_t*)(grey_image + i + j*width)) = _mm_extract_epi32(grey_pixels, 0);
                }
            }

            stbi_write_jpg(grey_image_filename, width, height, 1, grey_image, 10);
            free(rgb_image);

            gettimeofday(&fin,NULL);

            printf("Tiempo: %f\n", ((fin.tv_sec*1000000+fin.tv_usec)-(ini.tv_sec*1000000+ini.tv_usec))*1.0/1000000.0);
            free(grey_image_filename);
        }
    }
