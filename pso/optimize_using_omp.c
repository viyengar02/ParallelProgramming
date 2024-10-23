/* Implementation of PSO using OpenMP.
 *
 * Author: Naga Kandasamy
 * Date: May 10, 2023
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "pso.h"

int optimize_using_omp(char *function, int dim, int swarm_size, float xmin, float xmax, int num_iter, int num_threads){
    swarm_t *swarm;
    srand(time(NULL)); 
	swarm = pso_init(function, dim, swarm_size, xmin, xmax, num_threads); //init pso
    if (swarm == NULL) {
        fprintf(stderr, "Unable to initialize PSO\n");
        exit(EXIT_FAILURE);
    }

    int g = pso_solve_omp(function, swarm, xmax, xmin, num_iter, num_threads); //solve pso
    if (g >= 0) {
        fprintf(stderr, "Solution:\n");
        pso_print_particle(&swarm->particle[g]);
    }

    pso_free(swarm);
    return g;
}

int pso_solve_omp(char *function, swarm_t *swarm, float xmax, float xmin, int max_iter, int num_threads){
    int i, j, iter, g;
    float w, c1, c2;
    float r1, r2;
    float curr_fitness;
    particle_t *particle, *gbest;
	//static unsigned seed = 1;
    w = 0.79;
    c1 = 1.49;
    c2 = 1.49;
    iter = 0;
    g = -1;
	int swarm_size = swarm->num_particles;
	
	
    while (iter < max_iter) {
#pragma omp parallel for private(i, j, r1, r2, particle, gbest, curr_fitness) shared(swarm) num_threads(num_threads)
        for (i = 0; i < swarm_size; i++) {
            particle = &swarm->particle[i];
            gbest = &swarm->particle[particle->g]; //from last iteration
            for (j = 0; j < particle->dim; j++) { // update state
             unsigned int seed = (unsigned int)(seed + omp_get_thread_num());
                r1 = (float)rand_r(&seed) / (float)RAND_MAX;
                r2 = (float)rand_r(&seed) / (float)RAND_MAX;

                // update velocity
                particle->v[j] = w * particle->v[j] + c1 * r1 * (particle->pbest[j] - particle->x[j]) + c2 * r2 * (gbest->x[j] - particle->x[j]);

                if ((particle->v[j] < -fabsf(xmax - xmin)) || (particle->v[j] > fabsf(xmax - xmin))) {
                    particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));
                }

                // update  position
                particle->x[j] = particle->x[j] + particle->v[j];
                if (particle->x[j] > xmax) {
                    particle->x[j] = xmax;
                }
                if (particle->x[j] < xmin) {
                    particle->x[j] = xmin;
                }
            }

            pso_eval_fitness(function, particle, &curr_fitness);

            // update pbest
            if (curr_fitness < particle->fitness) {
                particle->fitness = curr_fitness;
                for (j = 0; j < particle->dim; j++) {
                    particle->pbest[j] = particle->x[j];
                }
            }
        }

        g = pso_get_best_fitness(swarm, num_threads); 

#pragma omp parallel for private(i) shared(swarm) num_threads(num_threads)
        for (i = 0; i < swarm_size; i++) {
            particle = &swarm->particle[i];
            particle->g = g;
        }

        #ifdef SIMPLE_DEBUG
        /* Print best performing particle */
        fprintf(stderr, "\nIteration %d:\n", iter);
        pso_print_particle(&swarm->particle[g]);
        #endif

        iter++;
    } /* End of iteration */

    return g;
}