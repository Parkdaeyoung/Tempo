#include <omp.h>
int x, y, z[1000];
#pragma omp threadprivate(x)

void default_none(int a) {
	const int c = 1;
	int i = 0;

#pragma omp parallel default(none) private(a) shared(z, c)
	{
		int j = omp_get_num_threads();
		/* O.K. - j is declared within parallel region */
		a = z[j];	/* O.K - a is listed in private clause */
					/*	   - z is listed in shared cluase */
		x = c;		/* O.K - x is thread-private */
					/*	   - c has const-qualified type and
 					 *	   	 is listed in shared cluase */
		z[i] = y;	/* Error - cannot reference i or y here */
	}

}
