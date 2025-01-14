#include "types.h"
#include "config.h"
extern "C"
{
#include "my_timers.h"
}
#ifdef USE_CUDA

#define PARTICLES_PER_THREAD (NUM_PARTICLES / (NUM_BLOCKS * NUM_THREADS))

static Particle *cudaInput = NULL;
static Particle **cudaOutput = NULL;

__device__ static unsigned int cudaRand(unsigned int *state)
{
    *state = *state * 1103515245 + 12345;
    return (unsigned int)(*state / (RAND_MAX)) % (RAND_MAX >> 1);    
}

__device__ static REAL randr(unsigned int *state) 
{
    REAL r = SIN(cudaRand(state) * cudaRand(state));
    return RANDOM_LIMIT[0] + (RANDOM_LIMIT[1] - RANDOM_LIMIT[0]) * ABS(r);
}

__device__ static REAL distance(const Vec3 *first, const Vec3 *second)
{
    return SQRT((first->x - second->x) * (first->x - second->x)
            +   (first->y - second->y) * (first->y - second->y)
            +   (first->z - second->z) * (first->z - second->z));
}

__global__ void cudaGenerate(Particle *cudaInput, unsigned int seed)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t shift = index * PARTICLES_PER_THREAD;
    unsigned int state = seed + index;

    for(size_t i = 0; i < PARTICLES_PER_THREAD; i++)
    {
        cudaInput[shift + i].x = randr(&state);
        cudaInput[shift + i].y = randr(&state);
        cudaInput[shift + i].z = randr(&state);
    }
}

__global__ void cudaFind(Particle *cudaInput, Particle **cudaOutput, int *sum)
{
    Vec3 center = CENTER;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t shift = index * PARTICLES_PER_THREAD;
    int count = 0;

    for(size_t i = 0; i < PARTICLES_PER_THREAD; i++)
    {
        if(ABS(distance(&(cudaInput[shift + i]), &center) - DISTANCE) <= DELTA)
        {
            cudaOutput[shift + count++] = &(cudaInput[shift + i]);
        }
    }

    atomicAdd(sum, count);
}

extern "C"
{

bool Generate(Particle *buffer, void *context)
{
    cudaMalloc(&cudaInput, NUM_PARTICLES * sizeof(Particle));

    cudaGenerate<<<NUM_BLOCKS, NUM_THREADS>>>(cudaInput, start_time());

    cudaMemcpy(buffer, cudaInput, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    return true;
}

size_t Find(Particle *input, Particle **output, void *context)
{
    int *deviceSum;
    int sum = 0;
    cudaMalloc(&cudaOutput, NUM_PARTICLES * sizeof(Particle*));
    cudaMalloc(&deviceSum, sizeof(*deviceSum));
    
    cudaMemcpy(deviceSum, &sum, sizeof(sum), cudaMemcpyHostToDevice);

    cudaFind<<<NUM_BLOCKS, NUM_THREADS>>>(cudaInput, cudaOutput, deviceSum);

    cudaMemcpy(output, cudaOutput, NUM_PARTICLES * sizeof(Particle*), cudaMemcpyDeviceToHost);
    cudaFree(cudaOutput);
    cudaFree(cudaInput);

    cudaMemcpy(&sum, deviceSum, sizeof(sum), cudaMemcpyDeviceToHost);
    return sum;
}

bool Initialize(int argc, char **argv, void **context)
{
    return true;
}

void Finalize(void *context)
{
    
}

}

#if (NUM_PARTICLES % (NUM_BLOCKS * NUM_THREADS))
    #error Number of particles must be divisible by the number of blocks times the number of threads
#endif

#endif
