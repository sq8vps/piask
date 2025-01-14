#ifndef CONFIG_H_
#define CONFIG_H_

//optimization method to use
//define here or using -D in compiler command line
//#define USE_PLAIN
//#define USE_OPENMP
//#define USE_MPI

#define NUM_THREADS 4 //number of threads to be used

//type to use for real numbers
#define USE_FLOAT
//#define USE_DOUBLE
//#define USE_LONG_DOUBLE

#define NUM_PARTICLES 10000000 //number of particles
#define CENTER {0.L, 0.L, 0.L} //center point
#define DISTANCE ((REAL)1.L) //distance from the center point
#define DELTA ((REAL)0.01L) //max. discrepancy when comparing distance (to compensate for rounding errors)
#define RANDOM_LIMIT (REAL[2]){-10.L, 10.L} //random coordinates generation lower and higher limit


#if !(defined(USE_PLAIN) ^ defined(USE_OPENMP) ^ defined(USE_MPI))
#error One and only one optimization method might be used
#endif

#if !(defined(USE_FLOAT) ^ defined(USE_DOUBLE) ^ defined(USE_LONG_DOUBLE))
#error One and only one real number type might be used
#endif


#endif