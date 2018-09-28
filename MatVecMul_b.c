void MatVecMul(double *c, double *A, double *x, double *b, int N) {

	// 1. c = Ax
#pragma omp target teams distribute parallel for \
	map(to:A[0:N*N], x[0:N])\
   	map(from:c[0:N]) 
	for (int i = 0; i < N; ++i) {
		double sum = 0.0;
		for (int j = 0; j < N; ++j) {
			sum += A[i*N+j] * x[j];
		}
		c[i] = sum;
	}

	// 2. c = c + b
#pragma omp target teams distribute parallel for\
   	map(tofrom:c[0:N])\
   	map(to:b[0:N])
	for (int i = 0; i < N; ++i) {
		c[i] = c[i] + b[i];
	}
}
