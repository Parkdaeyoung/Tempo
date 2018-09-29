int main()
{
	int N;
	int sum;
	N = 1024;
	sum = 0;
#pragma omp target teams distribute parallel for collapse(2)
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			sum += i * j;
		}
	}


	return 0;
}
