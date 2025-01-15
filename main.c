#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "my_timers.h"
#include "config.h"
#include "types.h"
#include "algo.h"

static Particle particles[NUM_PARTICLES];
static Particle *found[NUM_PARTICLES];
static unsigned int count = 0;

int main(int argc, char *argv[])
{
    srand(start_time());

    void *ctx = NULL;
    if(Initialize(argc, argv, &ctx))
    {
#if defined(USE_FLOAT)
        printf("Using float for real numbers\n");
#elif defined(USE_DOUBLE)
        printf("Using double for real numbers\n");
#elif defined(USE_LONG_DOUBLE)
        printf("Using long double for real numbers\n");
#endif
        printf("Number of particles: %d\n", NUM_PARTICLES);
        printf("Center: (%Lf, %Lf, %Lf)\n", (long double[3])CENTER[0], (long double[3])CENTER[1], (long double[3])CENTER[2]);
        printf("Distance: %Lf, max. discrepancy: %Lf\n", (long double)DISTANCE, (long double)DELTA);
        printf("Generating random particles with coordinates from %Lf to %Lf... ", (long double)RANDOM_LIMIT[0], (long double)RANDOM_LIMIT[1]);
        fflush(stdout);
    }

    if(Generate(particles, ctx))
    {
        printf("OK\n");

        printf("Finding particles at a given distance from the center... ");
        fflush(stdout);
    
        start_time();
    }

    if(Find(particles, found, ctx, &count))
    {
        stop_time();

        printf("OK\n");

        printf("Found %u particles in %f ms\n", count, elapsed_time());
    }
    
    Finalize(ctx);

    return 0;
}
