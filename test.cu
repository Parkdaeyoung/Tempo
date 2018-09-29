__global__ void vecAdd( int *C, int *B, int *A, int N=1024)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}
