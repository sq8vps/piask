#include "types.h"
#include "config.h"

#ifdef USE_CUDA

extern "C"
{
#include "my_timers.h"
}
#include <curand.h>
#include <curand_kernel.h>

#define PARTICLES_PER_THREAD (NUM_PARTICLES / (NUM_BLOCKS * NUM_THREADS))

static Particle *cudaInput = NULL;
static Particle **cudaOutput = NULL;

__device__ static REAL randr(curandState_t *state) 
{
   return curand_uniform(state) * (RANDOM_LIMIT[1] - RANDOM_LIMIT[0]) + RANDOM_LIMIT[0];
	// REAL r = SIN(cudaRand(state) * cudaRand(state));
   // return RANDOM_LIMIT[0] + (RANDOM_LIMIT[1] - RANDOM_LIMIT[0]) * ABS(r);
}

__device__ static REAL distance(const Vec3 *first, const Vec3 *second)
{
    return SQRT((first->x - second->x) * (first->x - second->x)
            +   (first->y - second->y) * (first->y - second->y)
            +   (first->z - second->z) * (first->z - second->z));
}

__global__ static void initRandom(curandState_t *state, unsigned long long seed)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed + index, index, 0, &(state[index]));
}

__global__ void cudaGenerate(Particle *cudaInput, curandState_t *state)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t shift = index * PARTICLES_PER_THREAD;

    for(size_t i = 0; i < PARTICLES_PER_THREAD; i++)
    {
        cudaInput[shift + i].x = randr(&(state[index]));
        cudaInput[shift + i].y = randr(&(state[index]));
        cudaInput[shift + i].z = randr(&(state[index]));
    }
}

__global__ void cudaFind(Particle *cudaInput, Particle **cudaOutput, unsigned int *sum)
{
    Vec3 center = CENTER;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t shift = index * PARTICLES_PER_THREAD;
    unsigned int count = 0;

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
    curandState_t *state;
    cudaMalloc(&state, NUM_BLOCKS * NUM_THREADS * sizeof(*state));

    initRandom<<<NUM_BLOCKS, NUM_THREADS>>>(state, start_time());
    cudaGenerate<<<NUM_BLOCKS, NUM_THREADS>>>(cudaInput, state);

    cudaMemcpy(buffer, cudaInput, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    return true;
}

bool Find(Particle *input, Particle **output, void *context, unsigned int *sum)
{
    unsigned int *deviceSum;
    cudaMalloc(&cudaOutput, NUM_PARTICLES * sizeof(Particle*));
    cudaMalloc(&deviceSum, sizeof(*deviceSum));
    
    cudaMemcpy(deviceSum, sum, sizeof(*sum), cudaMemcpyHostToDevice);

    cudaFind<<<NUM_BLOCKS, NUM_THREADS>>>(cudaInput, cudaOutput, deviceSum);

    cudaMemcpy(output, cudaOutput, NUM_PARTICLES * sizeof(Particle*), cudaMemcpyDeviceToHost);
    cudaFree(cudaOutput);
    cudaFree(cudaInput);

    cudaMemcpy(sum, deviceSum, sizeof(*sum), cudaMemcpyDeviceToHost);
    cudaFree(deviceSum);
    return true;
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
