#include "algo.h"
#include "config.h"
#include <stdlib.h>

#ifdef USE_MPI

#include <mpi.h>

#if defined(USE_FLOAT)
    #define MPI_TYPE MPI_FLOAT
#elif defined(USE_DOUBLE)
    #define MPI_TYPE MPI_DOUBLE
#elif defined(USE_LONG_DOUBLE)
    #define MPI_TYPE MPI_LONG_DOUBLE
#endif

struct MpiContext
{
    int size;
    int rank;
    MPI_Datatype type;
    Particle *input;
    Particle **output;
};

bool Generate(Particle *buffer, void *context)
{
    struct MpiContext *mpi = context;

    mpi->input = malloc(sizeof(Particle) * NUM_PARTICLES / mpi->size);

    for(size_t i = 0; i < NUM_PARTICLES / mpi->size; i++)
    {
        mpi->input[i].x = randr();
        mpi->input[i].y = randr();
        mpi->input[i].z = randr();
    }

    MPI_Gather(mpi->input, NUM_PARTICLES / mpi->size, mpi->type,
            buffer, NUM_PARTICLES / mpi->size, mpi->type,
            0, MPI_COMM_WORLD);
    return !(mpi->rank);
}

bool Find(Particle *input, Particle **output, void *context, unsigned int *sum)
{
    struct MpiContext *mpi = context;

    mpi->output = calloc(NUM_PARTICLES / mpi->size, sizeof(Particle*));

    MPI_Scatter(input, NUM_PARTICLES / mpi->size, mpi->type,
                mpi->input, NUM_PARTICLES / mpi->size, mpi->type,
                0, MPI_COMM_WORLD);

    unsigned int count = 0;
    Vec3 center = CENTER;

    for(size_t i = 0; i < NUM_PARTICLES / mpi->size; i++)
    {
        if(ABS(distance(&(mpi->input[i]), &center) - DISTANCE) <= DELTA)
        {
            mpi->output[count++] = &(mpi->input[i]);
        }
    }

    MPI_Reduce(&count, sum, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Gather(mpi->output, NUM_PARTICLES / mpi->size, MPI_UINT64_T,
            output, NUM_PARTICLES / mpi->size, MPI_UINT64_T,
            0, MPI_COMM_WORLD);

    return !mpi->rank;
}

bool Initialize(int argc, char **argv, void **context)
{
    MPI_Init(&argc, &argv);

    *context = malloc(sizeof(struct MpiContext));
    struct MpiContext *mpi = *context;

    MPI_Comm_size(MPI_COMM_WORLD, &(mpi->size));
    MPI_Comm_rank(MPI_COMM_WORLD, &(mpi->rank));

    MPI_Type_create_struct(3, 
        (int[3]){1, 1, 1}, 
        (MPI_Aint[3]){offsetof(Particle, x), offsetof(Particle, y), offsetof(Particle, z)}, 
        (MPI_Datatype[3]){MPI_TYPE, MPI_TYPE, MPI_TYPE}, &(mpi->type));
    MPI_Type_commit(&(mpi->type));
    return !(mpi->rank);
}

void Finalize(void *context)
{
    struct MpiContext *mpi = context;
    free(mpi->input);
    free(mpi->output);
    free(context);
    MPI_Finalize();
}
#endif
