/* Host code that implements a  separable convolution filter of a 
 * 2D signal with a gaussian kernel.
 * 
 * Author: Naga Kandasamy
 * Date modified: May 26, 2020
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

extern "C" void compute_gold(float *, float *, int, int, int);
extern "C" float *create_kernel(float, int);
void print_kernel(float *, int);
void print_matrix(float *, int, int);
void check_for_error(char *);

/* Width of convolution kernel */
#define HALF_WIDTH 8
#define COEFF 10
#define THREAD_BLOCK_SIZE 256
__constant__ float kernel_c[2 * HALF_WIDTH + 1];
/* Uncomment line below to spit out debug information */
//#define DEBUG

/* Include device code */
#include "separable_convolution_kernel.cu"
void check_for_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return;
}
/* function to compute the convolution on the device.*/
void compute_on_device_naive(float *gpu_result, float *matrix_c,\
		float *kernel, int num_cols,\
		int num_rows, int half_width)
{

	float *gpu_dev, *mat_dev, *kernel_dev, *rowtocol;

	int num_elements = num_rows * num_cols;
	int kernel_size = (2 * half_width) + 1;
	struct timeval start, stop;	
	cudaMalloc((void**)&mat_dev, num_elements * sizeof(float));
	cudaMemcpy(mat_dev, matrix_c, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&kernel_dev, kernel_size * sizeof(float));
	cudaMemcpy(kernel_dev, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&gpu_dev, num_elements * sizeof(float));
	cudaMalloc((void**)&rowtocol, num_elements * sizeof(float));

	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);
	int num_thread_blocks = (num_rows + thread_block.x - 1) / thread_block.x;
	dim3 grid(num_thread_blocks, 1);



	printf("Performing Naive Convolution\n");
	gettimeofday(&start, NULL);
	convolve_rows_kernel_naive<<<grid, thread_block>>>(rowtocol, mat_dev, kernel_dev, num_rows, num_cols, half_width);
	cudaDeviceSynchronize();

	check_for_error("KERNEL FAILURE");
	convolve_columns_kernel_naive<<<grid, thread_block>>>(gpu_dev, rowtocol, kernel_dev, num_rows, num_cols, half_width);
	cudaDeviceSynchronize();
	gettimeofday(&stop, NULL);

	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
				(stop.tv_usec - start.tv_usec)/(float)1000000));


	cudaMemcpy(gpu_result, gpu_dev, num_elements * sizeof (float), cudaMemcpyDeviceToHost);	
	cudaFree(gpu_dev);
	cudaFree(mat_dev);
	cudaFree(kernel_dev);
	cudaFree(rowtocol);
}

void compute_on_device_optimized(float *gpu_result, float *matrix_c,\
		float *kernel, int num_cols,\
		int num_rows, int half_width)
{
	float *gpu_dev, *mat_dev, *rowtocol;
	int num_elements = num_rows * num_cols;
	int kernel_size = (2 * half_width) + 1;
	struct timeval start, stop;
	cudaMalloc((void**)&mat_dev, num_elements * sizeof(float));
	cudaMemcpy(mat_dev, matrix_c, num_elements * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&gpu_dev, num_elements * sizeof(float));
	cudaMalloc((void**)&rowtocol, num_elements * sizeof(float));

	dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);
	int num_thread_blocks = (num_rows + thread_block.x - 1) / thread_block.x;
	dim3 grid(num_thread_blocks, 1);

	printf("Performing Optimized Convolution\n");
	cudaMemcpyToSymbol(kernel_c, kernel, kernel_size*sizeof(float));
	gettimeofday(&start, NULL);

	convolve_rows_kernel_optimized<<<grid, thread_block>>>(rowtocol, mat_dev, num_rows, num_cols, half_width);
	cudaDeviceSynchronize();
	convolve_columns_kernel_optimized<<<grid, thread_block>>>(gpu_dev, rowtocol, num_rows, num_cols, half_width);
	cudaDeviceSynchronize();
	gettimeofday(&stop, NULL);

	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
				(stop.tv_usec - start.tv_usec)/(float)1000000));

	cudaMemcpy(gpu_result, gpu_dev, num_elements * sizeof (float), cudaMemcpyDeviceToHost);
	cudaFree(gpu_dev);
	cudaFree(mat_dev);
	cudaFree(rowtocol);
}	


int main(int argc, char **argv)
{
	if (argc < 3) {
		printf("Usage: %s num-rows num-columns\n", argv[0]);
		printf("num-rows: height of the matrix\n");
		printf("num-columns: width of the matrix\n");
		exit(EXIT_FAILURE);
	}

	int num_rows = atoi(argv[1]);
	int num_cols = atoi(argv[2]);

	/* Create input matrix */
	int num_elements = num_rows * num_cols;
	printf("Creating input matrix of %d x %d\n", num_rows, num_cols);
	float *matrix_a = (float *)malloc(sizeof(float) * num_elements);
	float *matrix_c = (float *)malloc(sizeof(float) * num_elements);

	srand(time(NULL));
	int i;
	for (i = 0; i < num_elements; i++) {
		matrix_a[i] = rand()/(float)RAND_MAX;			 
		matrix_c[i] = matrix_a[i]; /* Copy contents of matrix_a into matrix_c */
	}

	/* Create Gaussian kernel */	  
	float *gaussian_kernel = create_kernel((float)COEFF, HALF_WIDTH);	
#ifdef DEBUG
	print_kernel(gaussian_kernel, HALF_WIDTH); 
#endif

	/* Convolve matrix along rows and columns. 
	   The result is stored in matrix_a, thereby overwriting the 
	   original contents of matrix_a.		
	 */
	printf("\nConvolving the matrix on the CPU\n");	 
	struct timeval start, stop;
	gettimeofday(&start, NULL);
	compute_gold(matrix_a, gaussian_kernel, num_cols,\
			num_rows, HALF_WIDTH);
	gettimeofday(&stop, NULL);
	printf("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
				(stop.tv_usec - start.tv_usec)/(float)1000000));
#ifdef DEBUG	 
	print_matrix(matrix_a, num_cols, num_rows);
#endif

	float *gpu_result_naive = (float *)malloc(sizeof(float) * num_elements);
	float *gpu_result_opt = (float *)malloc(sizeof(float) * num_elements);
	/* function to complete the functionality on the GPU.
	   The input matrix is matrix_c and the result must be stored in 
	   gpu_result.
	 */
	printf("\nConvolving matrix on the GPU\n");
	compute_on_device_naive(gpu_result_naive, matrix_c, gaussian_kernel, num_cols,\
			num_rows, HALF_WIDTH);

	compute_on_device_optimized(gpu_result_opt, matrix_c, gaussian_kernel, num_cols,\
			num_rows, HALF_WIDTH);

	printf("\nComparing CPU and GPU results\n");
	float sum_delta_naive = 0, sum_ref_naive = 0;
	for (i = 0; i < num_elements; i++) {
		sum_delta_naive += fabsf(matrix_a[i] - gpu_result_naive[i]);
		sum_ref_naive   += fabsf(matrix_a[i]);
	}

	float sum_delta_opt = 0, sum_ref_opt = 0;
	for (i = 0; i < num_elements; i++) {
		sum_delta_opt += fabsf(matrix_a[i] - gpu_result_opt[i]);
		sum_ref_opt   += fabsf(matrix_a[i]);
	}

	float L1norm_naive = sum_delta_naive / sum_ref_naive;
	float L1norm_opt = sum_delta_opt / sum_ref_opt;
	float eps = 1e-6;
	printf("L1 norm (naive): %E\n", L1norm_naive);
	printf((L1norm_naive < eps) ? "TEST PASSED\n" : "TEST FAILED\n");

	printf("L1 norm (optimized): %E\n", L1norm_opt);
	printf((L1norm_opt < eps) ? "TEST PASSED\n" : "TEST FAILED\n");


	free(matrix_a);
	free(matrix_c);
	free(gpu_result_naive);
	free(gpu_result_opt);
	free(gaussian_kernel);

	exit(EXIT_SUCCESS);
}



/* Print convolution kernel */
void print_kernel(float *kernel, int half_width)
{
	int i, j = 0;
	for (i = -half_width; i <= half_width; i++) {
		printf("%0.2f ", kernel[j]);
		j++;
	}

	printf("\n");
	return;
}

/* Print matrix */
void print_matrix(float *matrix, int num_cols, int num_rows)
{
	int i,  j;
	float element;
	for (i = 0; i < num_rows; i++) {
		for (j = 0; j < num_cols; j++){
			element = matrix[i * num_cols + j];
			printf("%0.2f ", element);
		}
		printf("\n");
	}

	return;
}
