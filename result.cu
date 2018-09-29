#include "/tmp/__o2c_kernel-583c0e.h"
void MatVecMul(double* c, double *A, double *x, double *b, int N) 
{
	int k;
	{
//#pragma omp target teams distribute parallel for map(to:A[N:N], x[0:N]) map(from:c[0:N])
{
//  Make up data environment

// Initial Memory Management
int created_A;
double * __o2c_device_A = CreateOrGetBuffer(&A[N], N * sizeof(A[N]), 1, &created_A);
int created_x;
double * __o2c_device_x = CreateOrGetBuffer(&x[0], N * sizeof(x[0]), 1, &created_x);
int created_c;
double * __o2c_device_c = CreateOrGetBuffer(&c[0], N * sizeof(c[0]), 0, &created_c);


{


// team configuration
PushNumTeams();
PushThreadLimit();


		{
SetArgument(&N, 0);
SetArgument(&A, 1);
SetArgument(&x, 2);
SetArgument(&c, 3);
LaunchKernel();
}


// team configuration clean up
PopNumTeams();
PopThreadLimit();

}

// Memory cleanup
DestroyBuffer(&__o2c_device_A, N * sizeof(A[N]), 1, created_A);
DestroyBuffer(&__o2c_device_x, N * sizeof(x[0]), 1, created_x);
DestroyBuffer(&__o2c_device_c, N * sizeof(c[0]), 0, created_c);

}


//#pragma omp target teams distribute parallel for map(tofrom:c[0:N]) map(to:b[0:N])
{
//  Make up data environment

// Initial Memory Management
int created_c;
double * __o2c_device_c = CreateOrGetBuffer(&c[0], N * sizeof(c[0]), 1, &created_c);
int created_b;
double * __o2c_device_b = CreateOrGetBuffer(&b[0], N * sizeof(b[0]), 1, &created_b);


{


// team configuration
PushNumTeams();
PushThreadLimit();


		{
SetArgument(&N, 0);
SetArgument(&c, 1);
SetArgument(&b, 2);
LaunchKernel();
}


// team configuration clean up
PopNumTeams();
PopThreadLimit();

}

// Memory cleanup
DestroyBuffer(&__o2c_device_c, N * sizeof(c[0]), 1, created_c);
DestroyBuffer(&__o2c_device_b, N * sizeof(b[0]), 1, created_b);

}

	}
}
#include <cuda_runtime.h>
__global__ void KernelName(int N, double * A, double * x, double * c)
{
/*
#pragma omp target teams distribute parallel for map(to: A[N:N],x[0:N]) map(from: c[0:N])
    for (int i = 0; i < N; ++i) {
        double sum = 0.;
        for (int j = 0; j < N; ++j) {
            sum += A[i * N + j] * x[j];
        }
        c[i] = sum;
    }

*/
int __o2c_gid = blockDim.x * blockIdx.x + threadIdx.x;
int __o2c_gsize = blockDim.x * gridDim.x;
i = 0;
for (int __o2c_i = __o2c_gid; 
__o2c_i <= (N - 0 - 1 + 1) / 1 - 1; 
__o2c_i += __o2c_gsize) {
i = 0 + (.omp.iv) * 1;
{
    double sum = 0.;
    for (int j = 0; j < N; ++j) {
        sum += A[i * N + j] * x[j];
    }
    c[i] = sum;
}

}

}
__global__ void KernelName(int N, double * c, double * b)
{
/*
#pragma omp target teams distribute parallel for map(tofrom: c[0:N]) map(to: b[0:N])
    for (int i = 0; i < N; ++i) {
        c[i] = c[i] + b[i];
    }

*/
int __o2c_gid = blockDim.x * blockIdx.x + threadIdx.x;
int __o2c_gsize = blockDim.x * gridDim.x;
i = 0;
for (int __o2c_i = __o2c_gid; 
__o2c_i <= (N - 0 - 1 + 1) / 1 - 1; 
__o2c_i += __o2c_gsize) {
i = 0 + (.omp.iv) * 1;
{
    c[i] = c[i] + b[i];
}

}

}
