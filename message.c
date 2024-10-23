/* Implementation of the SAXPY loop.
 *
 * Compile as follows: gcc -o saxpy saxpy.c -O3 -Wall -std=c99 -lpthread -lm
 *
 * Author: Naga Kandasamy
 * Date created: April 14, 2020
 * Date modified: April 13, 2023 
 *
 * Student names: Varun Iyengar and Cassius Garcia
 * Date: 4/24/2022
 *
 * */

#define _REENTRANT /* Make sure library functions are MT (muti-thread) safe */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

/* Function prototypes */
void compute_gold(float *, float *, float, int);
void compute_gold_t(void *);
void compute_using_pthreads_v1(float *, float *, float, int, int);
void compute_using_pthreads_v2(float *, float *, float, int, int);
int check_results(float *, float *, int, float);

/* Structure used to pass arguments to the worker threads*/
typedef struct args_for_thread_t {
    int tid;                /* Thread ID */
    float *arg1;               /* First argument */
    float *arg2;             /* Second argument */
    float arg3;               /* Third argument */
    int arg4;                  /* fourth argument */
    int arg5;                  /* fifth argument */
} ARGS_FOR_THREAD; 

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s num-elements num-threads\n", argv[0]);
        fprintf(stderr, "num-elements: Number of elements in the input vectors\n");
        fprintf(stderr, "num-threads: Number of threads\n");
		exit(EXIT_FAILURE);
	}

    pthread_t main_thread;
    main_thread = pthread_self(); /* Returns the thread ID for the calling thread */
    printf("Main thread = %lu\n", main_thread);
	
    int num_elements = atoi(argv[1]); 
    int num_threads = atoi(argv[2]);

	/* Create vectors X and Y and fill them with random numbers between [-.5, .5] */
    fprintf(stderr, "Generating input vectors\n");
    int i;
	float *x = (float *)malloc(sizeof(float) * num_elements);
    float *y1 = (float *)malloc(sizeof(float) * num_elements);              /* For the reference version */
	float *y2 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 1 */
	float *y3 = (float *)malloc(sizeof(float) * num_elements);              /* For pthreads version 2 */

	srand(time(NULL)); /* Seed random number generator */
	for (i = 0; i < num_elements; i++) {
		x[i] = rand()/(float)RAND_MAX - 0.5;
		y1[i] = rand()/(float)RAND_MAX - 0.5;
        y2[i] = y1[i]; /* Make copies of y1 for y2 and y3 */
        y3[i] = y1[i]; 
	}

    float a = 2.5;  /* Choose some scalar value for a */

	/* Calculate SAXPY using the reference solution. The resulting values are placed in y1 */
    fprintf(stderr, "\nCalculating SAXPY using reference solution\n");
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	
    compute_gold(x, y1, a, num_elements); 
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Compute SAXPY using pthreads, version 1. Results must be placed in y2 */
    fprintf(stderr, "\nCalculating SAXPY using pthreads, version 1\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v1(x, y2, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Compute SAXPY using pthreads, version 2. Results must be placed in y3 */
    fprintf(stderr, "\nCalculating SAXPY using pthreads, version 2\n");
	gettimeofday(&start, NULL);

	compute_using_pthreads_v2(x, y3, a, num_elements, num_threads);
	
    gettimeofday(&stop, NULL);
	fprintf(stderr, "Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec)/(float)1000000));

    /* Check results for correctness */
    fprintf(stderr, "\nChecking results for correctness\n");
    float eps = 1e-12;                                      /* Do not change this value */
    if (check_results(y1, y2, num_elements, eps) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else 
        fprintf(stderr, "TEST FAILED\n");
 
    if (check_results(y1, y3, num_elements, eps) == 0)
        fprintf(stderr, "TEST PASSED\n");
    else 
        fprintf(stderr, "TEST FAILED\n");

	/* Free memory */ 
	free((void *)x);
	free((void *)y1);
    free((void *)y2);
	free((void *)y3);
    printf("Main thread exiting\n");
    
    pthread_exit((void *)main_thread);
    exit(EXIT_SUCCESS);
}

/* Compute reference soution using a single thread */
void compute_gold(float *x, float *y, float a, int num_elements)
{
	int i;
    for (i = 0; i < num_elements; i++)
        y[i] = a * x[i] + y[i]; 
}

void compute_gold_t(void *this_arg)
{
    ARGS_FOR_THREAD *args_for_me = (ARGS_FOR_THREAD *)this_arg;
	int i;
    for (i = args_for_me->arg4; i < args_for_me->arg5; i++){
        args_for_me->arg2[i] = args_for_me->arg3 * args_for_me->arg1[i] + args_for_me->arg2[i]; 
    }
}

void *saxpyStride(void *args){

    ARGS_FOR_THREAD  *args_for_me = (ARGS_FOR_THREAD *)args;
    int tid = args_for_me->tid; 
    int stride = args_for_me->arg5; 

    while (tid < args_for_me->arg4) { 
        args_for_me->arg2[tid] = (args_for_me->arg3 * args_for_me->arg1[tid]) + args_for_me->arg2[tid]; 
        tid += stride; 
    }

    free((void *)args_for_me); 
    pthread_exit(NULL); 
}
/* Calculate SAXPY using pthreads, version 1. Place result in the Y vector */
void compute_using_pthreads_v1(float *x, float *y, float a, int num_elements, int num_threads)
{
    pthread_t *v1_thread = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ARGS_FOR_THREAD *args_for_thread;
    int i, size = floor(num_elements/num_threads);
    for (i = 0; i < num_threads; i++) {
        args_for_thread = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD)); /* Memory for structure to pack the arguments */
        args_for_thread->tid = i; /* Fill the structure with some dummy arguments */
        args_for_thread->arg1 = x; 
        args_for_thread->arg2 = y;
        args_for_thread->arg3 = a;
        args_for_thread->arg4 = i*size;
        args_for_thread->arg5 = args_for_thread->arg4 + size;
        
		
        if((pthread_create(&v1_thread[i], NULL, compute_gold_t, (void *)args_for_thread)) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }

    /* Join point: wait for all the worker threads to finish */
    for (i = 0; i < num_threads; i++){
        pthread_join(v1_thread[i], NULL);
    }
    /* Free data structures */
    free((void *)v1_thread);
    //pthread_exit((void *) v1_thread);
}

/* Calculate SAXPY using pthreads, version 2. Place result in the Y vector */
void compute_using_pthreads_v2(float *x, float *y, float a, int num_elements, int num_threads)
{
    	ARGS_FOR_THREAD *threadData;
	pthread_t *v2_thread = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

    for(int i = 0; i < num_threads; i++){ 
        threadData = (ARGS_FOR_THREAD *)malloc(sizeof(ARGS_FOR_THREAD)); 
        threadData -> tid = i;
        threadData -> arg5 = num_threads;
        threadData -> arg4 = num_elements;
        threadData -> arg3 = a;
        threadData -> arg1 = x;
        threadData -> arg2 = y;

        if(pthread_create(&v2_thread[i], NULL, saxpyStride, (void *)threadData) != 0){  
            perror("pthread_create failed");
        }
    }

    for (int j = 0; j < num_threads; j++){
        pthread_join(v2_thread[j], NULL); 
    }

}

/* Perform element-by-element check of vector if relative error is within specified threshold */
int check_results(float *A, float *B, int num_elements, float threshold)
{
    int i;
    for (i = 0; i < num_elements; i++) {
        if (fabsf((A[i] - B[i])/A[i]) > threshold)
            return -1;
    }
    
    return 0;
}
