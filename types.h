#ifndef TYPES_H_
#define TYPES_H_

#include "config.h"

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

typedef Vec3 Particle;

#endif