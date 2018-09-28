#include <omp.h>
#include <stdio.h>

int main()
{
	int flag=0;

#pragma omp parallel num_threads(3)
	{
		if (omp_get_thread_num() == 0)
		{
			/* set flag to release thread 1 */
#pragma omp atomic update
			flag++;
			/* Flush or flag is implied by the atomic directive */
		}
		else if (omp_get_thread_num() == 1)
		{
			/* Loop until we see that flag reaches 1 */
#pragma omp flush(flag)
			while(flag < 1)
			{
#pragma omp flsuh(flag)
			}
			printf("Thread 1 awoken\n");

			/* set flag to release thread 2 */
#pragma omp atomic update
			flag++;
			/* Flush of flag is implied by the atomic directive */
		}
		else if (omp_get_thread_num() == 2)
		{
			/* Loop until we see that flag reaches 2 */
#pragma omp flush(flag)
			while (flag < 2)
			{
#pragma omp flush(flag)
			}
			printf("Thread 2 awoken\n");
		}
	}

	return 0;
}
