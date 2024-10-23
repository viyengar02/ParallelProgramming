#include <sys/time.h>

/* file to complete the functionality of 2D separable 
 * convolution on the GPU. You may add additional kernel functions 
 * as necessary. 
 */

__global__ void convolve_rows_kernel_naive(float *result, float *input, float *kernel, int num_rows, int num_cols, int half_width)
{
	int i, ii, j, jj, q, x, y;
	y = blockIdx.x * blockDim.x + threadIdx.x;
	for (x = 0; x < num_cols; x++) {
		jj = x - half_width;
		q = x + half_width;
		if (jj < 0){
			jj = 0;}
		if (q >= num_cols){
			q = num_cols - 1;}
		ii = jj - x;
		jj = jj - x + half_width; 
		q = q - x + half_width;
		result[y * num_cols + x] = 0.0f;
		for(i = ii, j = jj; j <= q; j++, i++)
			result[y * num_cols + x] +=
				kernel[j] * input[y * num_cols + x + i];
	}
	return;
}

__global__ void convolve_columns_kernel_naive(float *result, float *input, float *kernel, int num_rows, int num_cols, int half_width)
{
	int i, ii, j, jj, q, x, y;
	y = blockIdx.x * blockDim.x + threadIdx.x;
	for(x = 0; x < num_cols; x++) {
		jj = y - half_width;
		q = y + half_width;
		if (jj < 0){jj = 0;}
		if (q >= num_rows){q = num_rows - 1;}
		ii = jj - y;
		jj = jj - y + half_width; 
		q = q - y + half_width;
		result[y * num_cols + x] = 0.0f;
		for (i = ii, j = jj; j <= q; j++, i++)
			result[y * num_cols + x] +=
				kernel[j] * input[y * num_cols + x + (i * num_cols)];
	}
	return; 
}

__global__ void convolve_rows_kernel_optimized(float *result, float *input, int num_rows, int num_cols, int half_width)
{
	int i, ii, j, jj, q, x, y;
	y = blockIdx.x * blockDim.x + threadIdx.x;
	for (x = 0; x < num_cols; x++) {
		jj = x - half_width;
		q = x + half_width;
		if (jj < 0){
			jj = 0;}
		if (q >= num_cols){
			q = num_cols - 1;}
		ii = jj - x;
		jj = jj - x + half_width; 
		q = q - x + half_width;
		result[y * num_cols + x] = 0.0f;
		for(i = ii, j = jj; j <= q; j++, i++)
			result[y * num_cols + x] +=
				kernel_c[j] * input[y * num_cols + x + i];
	}
	return; 
}

__global__ void convolve_columns_kernel_optimized(float *result, float *input, int num_rows, int num_cols, int half_width)
{
	int i, ii, j, jj, q, x, y;
	y = blockIdx.x * blockDim.x + threadIdx.x;
	for(x = 0; x < num_cols; x++) {
		jj = y - half_width;
		q = y + half_width;
		if (jj < 0){
			jj = 0;}
		if (q >= num_rows){
			q = num_rows - 1;}
		ii = jj - y;
		jj = jj - y + half_width; 
		q = q - y + half_width;
		result[y * num_cols + x] = 0.0f;
		for (i = ii, j = jj; j <= q; j++, i++)
			result[y * num_cols + x] +=
				kernel_c[j] * input[y * num_cols + x + (i * num_cols)];
	}
	return;
}