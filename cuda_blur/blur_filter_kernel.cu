/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void blur_filter_kernel (const float *in, float *out, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    int stride = blockDim.x * gridDim.x; //stride length 
    int curRow, curCol, num_neighbors; 
    float blur_val;

    while (tid < (size*size)){
        int row = tid/size;
        int col = tid % size;
        blur_val = 0.0;
        num_neighbors = 0;

        for (int i = -BLUR_SIZE; i < (BLUR_SIZE + 1); i++){
            for (int j = -BLUR_SIZE; j < (BLUR_SIZE + 1); j++){ //check neighbors/bounds
                curRow = row + i;
                curCol = col + j; 

                if ((curRow > -1) && (curRow < size) && (curCol > -1) && (curCol < size)) { //check bounds
                    blur_val += in[curRow * size + curCol]; 
                    num_neighbors += 1;
                }
            }
        }

        out[tid] = blur_val/num_neighbors; 
        tid += stride; 
    }

    return;
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
