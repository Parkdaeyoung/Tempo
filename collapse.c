#include <cuda_runtime.h>
#include "/tmp/__omp_tmp-50b7fa.h"
int main()
{

	int N = 1024;
	int sum;
//#pragma omp target teams distribute collapse(2)
{
//const int *const restrict .global_tid.
//const int *const restrict .bound_tid.
//struct (anonymous at collapse.c:6:9) *const restrict __context


// Initial Memory Management


{
//const int *const restrict .global_tid.
//const int *const restrict .bound_tid.
//struct (anonymous at collapse.c:6:9) *const restrict __context



// team configuration
PushNumTeams();
PushThreadLimit();


	KernelCall<<< 1, 2, 3, 4 >>>();


// team configuration clean up
PopNumTeams();
PopThreadLimit();

}

// Memory cleanup

}


	return 0;
}

