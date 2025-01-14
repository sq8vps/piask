#ifndef ALGO_H_
#define ALGO_H_

#include "types.h"
#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

static inline REAL randr(void) 
{
    REAL r = SIN(rand() * rand());
    return RANDOM_LIMIT[0] + (RANDOM_LIMIT[1] - RANDOM_LIMIT[0]) * ABS(r);
}

static inline REAL distance(const Vec3 *first, const Vec3 *second)
{
    return SQRT((first->x - second->x) * (first->x - second->x)
            +   (first->y - second->y) * (first->y - second->y)
            +   (first->z - second->z) * (first->z - second->z));
}


bool Generate(Particle *buffer, void *context);

size_t Find(Particle *input, Particle **output, void *context);

bool Initialize(int argc, char **argv, void **context);

void Finalize(void *context);

#endif