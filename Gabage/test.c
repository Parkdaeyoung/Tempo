#define N 1024
int main()
{

	double A[N*N], x[N], c[N];
#pragma omp target
	for (int i = 0; i < N; ++i) {
		double sum = 0.0;
		for (int j = 0; j < N; ++j) {
			sum += A[i*N+j] * x[j];
		}
		c[i] = sum;
	} 
}
