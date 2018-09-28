void foo()
{
	int A[30], *p;
#pragma omp target data map(A[0:4])
	{
		p = &A[0];
#pragma omp target map(p[7:20])
		{
			A[2] = 0;
			A[8] = 0;
		}
	}
}
