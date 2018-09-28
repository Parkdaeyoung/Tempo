void long_running_task(void);
void loop_body(int i, int j);

void parallel_work(void)
{
	int i, j;
#pragma omp taskgroup
	{
#pragma omp task
		long_running_task();
#pragma omp taskloop private(j) graainsize(500) nogroup
		for (i = 0; i < 10000; i++) { // can execute concurrently
			for (j = 0; j < i; j++) {
				loop_body(i, j);
			}
		}
	}
}
