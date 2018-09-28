void start( double *a, double *b, double *c, int n, int *ioff )
{
	int i;
#pragma omp simd
	for (i = 0; i < n; i++)
		a[i] *= b[i] * c[i + *ioff];
}
