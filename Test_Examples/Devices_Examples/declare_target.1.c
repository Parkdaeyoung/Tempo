#pragma omp declare target
extern void fib(int N);
#pragma omp end declare target
#define THRESHOLD 1000000
void fib_wrapper(int n)
{
#pragma omp target if(n>THRESHOLD)
	{
		fib(n);
	}
}
