#include <stdio.h>
void foo()
{
	int x = 2;
#pragma omp task mergeable
	{
		x++;
	} 
#pragma omp taskwait

	printf("%d\n", x); // prints 2 or 3
}
