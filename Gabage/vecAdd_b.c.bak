void vecAdd(double *A, double *B, double *C, int N) {
#pragma omp target map(to:A[0:N], B[0:N]) map(from:C[0:N]) map(to:N)
	{
#pragma omp teams num_teams(32) thread_limit(64)
		{
#pragma omp distribute parallel for
			for (int i = 0; i < N; ++i) {
				C[i] = A[i] + B[i];
			}
		}
	}
}
