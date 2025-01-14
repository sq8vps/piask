#include "algo.h"
#include "config.h"

#ifdef USE_OPENMP

#include <omp.h>

bool Generate(Particle *buffer, void *context)
{
    #pragma omp parallel for
    for(size_t i = 0; i < NUM_PARTICLES; i++)
    {
        buffer[i].x = randr();
        buffer[i].y = randr();
        buffer[i].z = randr();
    }
    return true;
}

size_t Find(Particle *input, Particle **output, void *context)
{
    size_t count = 0;
    Vec3 center = CENTER;

    #pragma omp parallel for schedule(guided) reduction(+:count)
    for(size_t i = 0; i < NUM_PARTICLES; i++)
    {
        if(ABS(distance(&(input[i]), &center) - DISTANCE) <= DELTA)
        {
            output[count++] = &(input[i]);
        }
    }
    return count;
}

bool Initialize(int argc, char **argv, void **context)
{
    omp_set_num_threads(NUM_THREADS);
    return true;
}

void Finalize(void *context)
{

}

#endif