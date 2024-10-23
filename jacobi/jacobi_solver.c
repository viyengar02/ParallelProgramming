/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: APril 26, 2023
 *
 * Student name(s): FIXME
 * Date modified: FIXME
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "jacobi_solver.h"

/* Uncomment the line below to spit out debug information */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s matrix-size num-threads\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
        fprintf(stderr, "num-threads: number of worker threads to create\n");
		exit(EXIT_FAILURE);
	}

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x_v1;      /* Solution computed by pthread code using chunking */
    matrix_t mt_solution_x_v2;      /* Solution computed by pthread code using striding */

	/* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x_v1 = allocate_matrix(matrix_size, 1, 0);
    mt_solution_x_v2 = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
    compute_gold(A, reference_x, B, max_iter);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	
	/* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x_v1 and mt_solution_x_v2.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using chunking\n");
	compute_using_pthreads_v1(A, mt_solution_x_v1, B, max_iter, num_threads);
    display_jacobi_solution(A, mt_solution_x_v1, B); /* Display statistics */
    
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using striding\n");
	compute_using_pthreads_v2(A, mt_solution_x_v2, B, max_iter, num_threads);
    display_jacobi_solution(A, mt_solution_x_v2, B); /* Display statistics */

    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x_v1.elements);
    free(mt_solution_x_v2.elements);
	
    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads using chunking. 
 * Result must be placed in mt_sol_x_v1. */
void compute_using_pthreads_v1(const matrix_t A, matrix_t mt_sol_x_v1, const matrix_t B, int max_iter, int num_threads)
{
	pthread_t *v1_thread = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ARGS_FOR_THREAD *args_for_thread;

    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_mutex_t lock;
    pthread_barrierattr_t barrier_attributes;
    pthread_barrier_t barrier;
    pthread_barrierattr_init(&barrier_attributes);
    pthread_barrier_init(&barrier, &barrier_attributes, num_threads);

    int converged = 0;
    float diff;
    int num_iter;
    int chunk = floor(mt_sol_x_v1.num_rows/num_threads);
    int rem = mt_sol_x_v1.num_rows % num_threads;
    pthread_mutex_init(&lock, NULL); 
    matrix_t new_x = allocate_matrix(A.num_rows, 1, 0)
;    
    int i;//, size = floor(max_iter/num_threads);
    for (i = 0; i < num_threads; i++) {
        args_for_thread = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD)); /* Memory for structure to pack the arguments */
        args_for_thread->tid = i; /* Fill the structure with some dummy arguments */
        args_for_thread->A = A; 
        args_for_thread->x = &mt_sol_x_v1;
        args_for_thread->new_x = &new_x;
        args_for_thread->B = B;
        args_for_thread->start = i*chunk;
        args_for_thread->stop = (i==num_threads-1) ? args_for_thread->start + chunk + rem : args_for_thread->start + chunk;
        args_for_thread->max_iter = max_iter;
        args_for_thread->num_threads = num_threads;
        args_for_thread->lock = &lock;
        args_for_thread->barrier = &barrier;
        args_for_thread->converged = &converged;
        args_for_thread->diff = &diff;
        
        args_for_thread->num_iter = &num_iter;
        
        if((pthread_create(&v1_thread[i], &attributes, compute_gold_v1, (void *)args_for_thread)) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }

    for (i = 0; i < num_threads; i++){
            pthread_join(v1_thread[i], NULL);
    }

    if (num_iter < max_iter)
        fprintf(stderr, "\nConvergence achieved after %d iterations\n", num_iter);
    else
        fprintf(stderr, "\nMaximum allowed iterations reached OR thread finished\n");

    pthread_mutex_destroy(&lock);
        /* Free data structures */
    free((void *)v1_thread);
}

void compute_gold_v1(void *args_for_me)
{
    // Initializations
    ARGS_FOR_THREAD *this_arg = (ARGS_FOR_THREAD *)args_for_me;
    int i, j;
    int tid = this_arg->tid;
    const matrix_t A = this_arg->A;
    const matrix_t B = this_arg->B;
    matrix_t *cur = this_arg->x, *prev = this_arg->new_x;
    matrix_t *tmp;
    int max_iter = this_arg->max_iter;
    int start = this_arg->start;
    int stop = this_arg->stop;
    int *num_iter = this_arg->num_iter; *num_iter = start;
    float *diff = this_arg->diff;
    int *converged = this_arg->converged;
    double mse;    
    /* Initialize current jacobi solution. */
    for (i = 0; i < A.num_rows; i++){
        prev->elements[i] = B.elements[i];
    }
    /* Perform Jacobi iteration. */
    while (!*converged) {
        if(tid==0){
            *diff = 0.0;
            (*num_iter)++;
            //printf("Initialization done\n");
        }
        pthread_barrier_wait(this_arg->barrier);  
        
        for (i = start; i < stop; i++) {
            double sum = 0.0;
            for (j = 0; j < A.num_columns; j++) {
                if (i != j){
                    sum += A.elements[i * A.num_columns + j] * prev->elements[j];
                }
            }
            //Update values for the unkowns for the current row.
            cur->elements[i] = (B.elements[i] - sum)/A.elements[i * A.num_columns + i];
        }
        //printf("Done calculating cur\n");
        /* Check for convergence and update the unknowns. */
        float pdiff=0.0;
        for (i = start; i < stop; i++) {
            pdiff += (cur->elements[i] - prev->elements[i]) * (cur->elements[i] - prev->elements[i]);
        }
        //printf("done pdiff\n"); 
        pthread_mutex_lock(this_arg->lock);
        *diff += pdiff;
        pthread_mutex_unlock(this_arg->lock);

        pthread_barrier_wait(this_arg->barrier);  
        mse = sqrt(*diff); /* Mean squared error. */

        //fprintf(stderr, "Thread %d Iteration: %d. MSE = %f\n", tid, *num_iter, mse); 

        if ((mse <= THRESHOLD) || (*num_iter == max_iter)){
            *converged = 1;
            //fprintf(stderr, "CONVERGED\n"); 
            for(i=start;i < stop;i++){
                this_arg->x->elements[i] = cur->elements[i];
            }
        }
        pthread_barrier_wait(this_arg->barrier);  
        tmp = prev;
        prev = cur;
        cur = tmp;
    }

    //free(prev->elements);
    pthread_exit(NULL);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads using striding. 
 * Result must be placed in mt_sol_x_v2. */
void compute_using_pthreads_v2(const matrix_t A, matrix_t mt_sol_x_v2, const matrix_t B, int max_iter, int num_threads)
{
	int i; //same code from saxpy to set up threads
    pthread_t *worker_thread = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ARGS_FOR_THREAD *thread_data;
	pthread_barrier_t *barrier = (pthread_barrier_t *)malloc(sizeof(pthread_barrier_t *));
	pthread_barrier_init(barrier,NULL,num_threads); //create and init barrier
	matrix_t new = allocate_matrix(mt_sol_x_v2.num_rows, 1, 0);
	double ssd[num_threads];
	memset( ssd, 0.0, num_threads*sizeof(double) );
    
    //Create worker threads
    for (i = 0; i < num_threads; i++) 
	{
        thread_data = (ARGS_FOR_THREAD *)malloc (sizeof(ARGS_FOR_THREAD));
        thread_data->tid = i;
        thread_data->num_threads = num_threads;
		thread_data->max_iter = max_iter;
		thread_data->barrier = barrier;
        thread_data->A = A;
        thread_data->B = B;
        thread_data->x = &mt_sol_x_v2;
		thread_data->new_x = &new;
		thread_data->ssd = ssd;
		

        if ((pthread_create(&worker_thread[i], NULL, jacobi_striding, (void *)thread_data)) != 0) {
            perror("pthread_create");
            return -1;
        }
    }

    //Joins worker threads 
    for (i = 0; i < num_threads; i++)
        pthread_join(worker_thread[i], NULL);
 
    return 0;
}

/* Multi-threaded implementation of AX = B using the concept of striding */
void *jacobi_striding(void *args)
{
    ARGS_FOR_THREAD *thread_data = (ARGS_FOR_THREAD *)args;
    int tid = thread_data->tid;
    int stride = thread_data->num_threads;
	int num_rows = thread_data->A.num_rows;
	int num_cols = thread_data->A.num_columns;
    double sum;
	double ssdsum ,mse;
	pthread_barrier_t *barrier = thread_data->barrier;
	matrix_t A = thread_data->A;
	matrix_t B = thread_data->B;
	matrix_t *ref = thread_data->x;
	matrix_t *new = thread_data->new_x;
	int max_iter = thread_data->max_iter;
	double* ssd = thread_data->ssd;
	int i,j;
	int done = 0;
	int num_iter = 0;
	
	matrix_t* prev; //data buffer
	matrix_t* cur;
	
	for (i = 0; i < num_rows; i++){
		ref->elements[i] = B.elements[i];
	}
	
	prev = ref;
	
    while(!done){
		if(num_iter % 2 == 0){
			cur = new;
		}else{
			cur = ref;
		}

		for(i = tid; i < num_rows; i += stride){ //striding
			sum = -A.elements[i * num_cols + i] * prev->elements[i];
			for(j = 0; j < num_cols; j++){
				sum += A.elements[i * num_cols + j] * prev->elements[j];
			}
			cur->elements[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
		}
			
		pthread_barrier_wait(barrier);
		ssd[tid] = 0.0;

		for (i = tid; i < num_rows; i += stride) {
			ssd[tid] += (cur->elements[i] - prev->elements[i]) * (cur->elements[i] - prev->elements[i]);
		}
		
		pthread_barrier_wait(barrier);
		ssdsum = 0.0;

		for (i = 0; i < stride; i++){
			ssdsum+=ssd[i];
		}

		pthread_barrier_wait(barrier);
		prev = cur;
		num_iter++;
		mse = sqrt(ssdsum); 

		if ((mse <= THRESHOLD) || (num_iter == max_iter)){
			done = 1;
		}
		
		pthread_barrier_wait(barrier);
	}

	if (num_iter < max_iter && tid == 0){
		fprintf(stderr, "%d iterations\n", num_iter);
	}else if(tid == 0){
		fprintf(stderr, "Max iterations\n");
	}
	
    free((void *)thread_data); //frees data
    pthread_exit(NULL);
}



/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
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

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}