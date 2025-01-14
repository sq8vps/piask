#include "algo.h"
#include "config.h"

#ifdef USE_CUDA

#define PARTICLES_PER_THREAD (NUM_PARTICLES / (NUM_BLOCKS * NUM_THREADS))

static Particle *cudaInput = NULL;
static Particle **cudaOutput = NULL;
__shared__ int sum;

__global__ void cudaGenerate(void)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t shift = index * PARTICLES_PER_THREAD;

    for(size_t i = 0; i < PARTICLES_PER_THREAD; i++)
    {
        cudaInput[shift + i].x = randr();
        cudaInput[shift + i].y = randr();
        cudaInput[shift + i].z = randr();
    }
}

__global__ void cudaFind(void)
{
    Vec3 center = CENTER;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    size_t shift = index * PARTICLES_PER_THREAD;

    int count = -1;

    for(size_t i = 0; i < PARTICLES_PER_THREAD; i++)
    {
        if(ABS(distance(&(cudaInput[shift + i]), &center) - DISTANCE) <= DELTA)
        {
            cudaOutput[shift + count++] = &(cudaInput[shift + i]);
        }
    }

    atomicAdd(&sum, count);
}

bool Generate(Particle *buffer, void *context)
{
    cudaMalloc((void**)&cudaInput, NUM_PARTICLES * sizeof(Particle));

    cudaGenerate<<<NUM_BLOCKS, NUM_THREADS>>>();

    cudaMemcpy(buffer, cudaInput, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    return true;
}

size_t Find(Particle *input, Particle **output, void *context)
{
    cudaMalloc((void***)&cudaOutput, NUM_PARTICLES * sizeof(Particle*));
    int t = 0;
    cudaMemcpy(&sum, &t, sizeof(sum), cudaMemcpyHostToDevice);

    cudaFind<<<NUM_BLOCKS, NUM_THREADS>>>();

    cudaMemcpy(output, cudaOutput, NUM_PARTICLES * sizeof(Particle*), cudaMemcpyDeviceToHost);
    cudaFree(cudaOutput);
    cudaFree(cudaInput);

    cudaMemcpy(&t, &sum, sizeof(sum), cudaMemcpyDeviceToHost);
    return t;
}

bool Initialize(int argc, char **argv, void **context)
{
    return true;
}

void Finalize(void *context)
{
    
}

#if (NUM_PARTICLES % (NUM_BLOCKS * NUM_THREADS))
    #error Number of particles must be divisible by the number of blocks times the number of threads
#endif

#endif
