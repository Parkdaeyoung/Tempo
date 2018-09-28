float dotprod(float B[], float C[], int N)
{
	float sum = 0;
	int i;
#pragma omp target teams map(to: B[0:N], C[0:N])\
	defaultmap(tofrom:scalar) reduction(+:sum)
#pragma omp distribute parallel for reduction(+:sum)
	for (i = 0; i < N; i++)
		sum += B[i] * C[i];
	return sum;
}
/* Note: The variables sum0, sum1 are now mapped with tofrom, for correct
 * execution with 4.5 (and pre-4.5) compliant compilers.
 * See Devices Intro.
 */
