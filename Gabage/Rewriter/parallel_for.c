void vecAdd(int *a, int *b, int *c, int N)
{
#pragma omp parallel 
	{
#pragma omp for
		{
			for (int i = 0; i < N; ++i) {
				c[i] = a[i] + b[i];
			}
		}
	}
}
