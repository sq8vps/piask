#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "my_timers.h"

//optimization method to use
#define USE_OPENMP

#define NUM_THREADS 4 //number of threads to be used

//type to use for real numbers
//#define USE_FLOAT
//#define USE_DOUBLE
#define USE_LONG_DOUBLE

#define NUM_PARTICLES 10000000 //number of particles
#define CENTER {0.L, 0.L, 0.L} //center point
#define DISTANCE ((REAL)1.L) //distance from the center point
#define DELTA ((REAL)0.01L) //max. discrepancy when comparing distance (to compensate for rounding errors)
#define RANDOM_LIMIT (REAL[2]){-10.L, 10.L} //random coordinates generation lower and higher limit

#if defined(USE_OPENMP)
#include <omp.h>
#endif

#if defined(USE_FLOAT)
    #define REAL float
    #define SQRT sqrtf
    #define ABS fabsf
    #define SIN sinf
#elif defined(USE_DOUBLE)
    #define REAL double
    #define SQRT sqrt
    #define ABS fabs
    #define SIN sin
#elif defined(USE_LONG_DOUBLE)
    #define REAL long double
    #define SQRT sqrtl
    #define ABS fabsl
    #define SIN sinl
#endif

typedef struct Vec3
{
    REAL x;
    REAL y; 
    REAL z;
} Vec3;

typedef struct Particle
{
    REAL mass;
    Vec3 position;
} Particle;

static Particle particles[NUM_PARTICLES];
static Particle *found[NUM_PARTICLES];
static size_t count = 0;

static inline REAL randr(void) 
{
    REAL r = SIN(rand() * rand());
    return RANDOM_LIMIT[0] + (RANDOM_LIMIT[1] - RANDOM_LIMIT[0]) * ABS(r);
}

static inline REAL getDistance(const Vec3 *first, const Vec3 *second)
{
    return SQRT((first->x - second->x) * (first->x - second->x)
            +   (first->y - second->y) * (first->y - second->y)
            +   (first->z - second->z) * (first->z - second->z));
}

int main(int argc, char *argv[])
{
    omp_set_num_threads(NUM_THREADS);

    srand(time(NULL));

    Vec3 center = CENTER;

#if defined(USE_FLOAT)
    printf("Using float for real numbers\n");
#elif defined(USE_DOUBLE)
    printf("Using double for real numbers\n");
#elif defined(USE_LONG_DOUBLE)
    printf("Using long double for real numbers\n");
#endif
    printf("Number of particles: %llu\n", NUM_PARTICLES);
    printf("Center: (%Lf, %Lf, %Lf)\n", (long double)center.x, (long double)center.y, (long double)center.z);
    printf("Distance: %Lf, max. discrepancy: %Lf\n", (long double)DISTANCE, (long double)DELTA);
    printf("Generating random particles with coordinates from %Lf to %Lf... ", (long double)RANDOM_LIMIT[0], (long double)RANDOM_LIMIT[1]);

#if defined(USE_OPENMP)
    #pragma omp parallel for
#endif
    for(size_t i = 0; i < NUM_PARTICLES; i++)
    {
        particles[i].position.x = randr();
        particles[i].position.y = randr();
        particles[i].position.z = randr();
    }

    printf("OK\n");

    printf("Finding particles at a given distance from the center... ");
    
    start_time();

#if defined(USE_OPENMP)
    #pragma omp parallel for schedule(guided) reduction(+:count)
#endif
    for(size_t i = 0; i < NUM_PARTICLES; i++)
    {
        if(ABS(getDistance(&(particles[i].position), &center) - DISTANCE) <= DELTA)
        {
            found[count++] = &(particles[i]);
        }
    }

    stop_time();

    printf("OK\n");

    printf("Found %llu particles in %f ms\n", count, elapsed_time());
    
    return 0;
}