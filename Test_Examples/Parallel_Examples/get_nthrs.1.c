#include <omp.h>
void work(int i);

void incorrect()
{
	int np, i;
	np = omp_get_num_threads(); /* misplaced */

#pragma omp parallel for schedule(static)
	for (i = 0; i < np; i++)
		work(i);
}
