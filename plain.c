#include "algo.h"
#include "config.h"

#ifdef USE_PLAIN

bool Generate(Particle *buffer, void *context)
{
    for(size_t i = 0; i < NUM_PARTICLES; i++)
    {
        buffer[i].x = randr();
        buffer[i].y = randr();
        buffer[i].z = randr();
    }
    return true;
}

bool Find(Particle *input, Particle **output, void *context, size_t *sum)
{
    *sum = 0;
    Vec3 center = CENTER;

    for(size_t i = 0; i < NUM_PARTICLES; i++)
    {
        if(ABS(distance(&(input[i]), &center) - DISTANCE) <= DELTA)
        {
            output[(*sum)++] = &(input[i]);
        }
    }
    return true;
}

bool Initialize(int argc, char **argv, void **context)
{
    return true;
}

void Finalize(void *context)
{
    
}

#endif