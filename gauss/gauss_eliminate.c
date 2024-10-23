/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date modified: April 26, 2023
 *
 * Student names(s): FIXME
 * Date: FIXME
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold_gauss.c -std=gnu99 -O3 -Wall -lpthread -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix, int);
void *elimination_thread(void *);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);



int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        fprintf(stderr, "num-threads: number of worker threads to create\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    int status = compute_gold(U_reference.elements, A.num_rows);
  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
  
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* FIXME: Perform Gaussian elimination using pthreads. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");
    
   
    gettimeofday(&start, NULL);

    gauss_eliminate_using_pthreads(U_mt, num_threads);
  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    fprintf(stderr, "multi-threaded Gaussian elimination was successful.\n");

    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}

typedef struct {
    Matrix U;
    int start_row;
    int end_row;
    int num_threads;
    pthread_mutex_t *lock; 
    pthread_barrier_t *barrier;
} ThreadArgs;

/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U, int num_threads)
{
    //ThreadArgs *thread_args;
    ThreadArgs thread_args[num_threads];
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
	pthread_barrier_t *barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t *));
    pthread_barrier_init(barrier, NULL, num_threads);
    pthread_mutex_t lock;
    pthread_mutex_init(&lock, NULL); 

    int num_rows = U.num_rows, start_row;
    int chunk_size = num_rows / num_threads;
    int remaining_rows = num_rows % num_threads;
    // Create and execute threads
    
    for (int i = 0; i < num_threads; i++) {
       start_row = i*chunk_size;
        //thread_args = (ThreadArgs *)malloc (sizeof(ThreadArgs));
        
        thread_args[i].U = U;
        thread_args[i].start_row = start_row;
        thread_args[i].end_row = (i==num_threads-1) ? start_row + chunk_size + remaining_rows : start_row + chunk_size;
        thread_args[i].barrier = barrier;
        thread_args[i].lock = &lock;
        
        if (pthread_create(&threads[i], &attributes, elimination_thread, (void *)&thread_args[i]) != 0) {
            fprintf(stderr, "Failed to create thread. Exiting.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Wait for threads to finish
    for (int i = 0; i < num_threads; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "Failed to join thread. Exiting.\n");
            exit(EXIT_FAILURE);
        }
    }

    pthread_mutex_destroy(&lock);
    free((void *)threads);
}

/* Function executed by each thread */
void *elimination_thread(void *arg)
{
    
    fprintf(stderr,"\nopened a thread\n");
    ThreadArgs *args = (ThreadArgs *)arg;
    Matrix U = args->U;
    int start_row = args->start_row;
    int end_row = args->end_row;
    int num_elements = U.num_columns;
    fprintf(stderr,"\nabout to go into loop\n");
    for (int k = start_row; k < end_row; k++) {
        for (int j = k+1; j < end_row; j++) {
            if (U.elements[num_elements * k + k] == 0) {
                fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                exit(EXIT_FAILURE);
            } 
            U.elements[num_elements*k+j] = (float) U.elements[num_elements * k + j] / U.elements[num_elements * k + k];
        }
        U.elements[num_elements * k + k] = 1;
        
        for (int i = k + 1+start_row; i < end_row; i++) {
            
            for (int j = k + 1; j < (num_elements); j++){
                U.elements[num_elements * i + j] = U.elements[num_elements * i + j] - (U.elements[num_elements * i + k] * U.elements[num_elements * k + j]);
            }
           
fprintf(stderr,"\nwaiting at barrier\n");
        pthread_barrier_wait(args->barrier);
        U.elements[num_elements * i + k] = 0;
        }
        
    }
         
    pthread_exit(NULL);
}

/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
  
    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}