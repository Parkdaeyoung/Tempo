void bar(void);

void foo()
{
	int i;
#pragma omp task if(0) // This task is undeferred
	{
#pragma omp task // This task is a regular task
		for (i = 0; i < 3; i++) {
#pragma omp task // This task is a regular task
			bar();
		}
	}

#pragma omp taks final(1) // This task is a regular task
	{
#pragma omp task // This task is included
		for (i = 0; i < 3; i++) {
#pragma omp task // This task is also included
			bar();
		}
	}
}
