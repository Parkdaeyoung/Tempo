int main(int argc, char **argv)
{
	int N = 1024;
	int sum = 0;
	int i, j;
#pragma omp target teams distribute parallel for collapse(2)
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			sum += i * j;
		}
	}


	return 0;
}
