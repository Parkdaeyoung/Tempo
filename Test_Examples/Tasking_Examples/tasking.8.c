int tp;
#pragma omp threadprivate(tp)
int var;

void work()
{
#pragma omp parallel
	{
		/* do work here */
#pragma omp task
		{
			tp++;
			/* do work here */
#pragma omp task
			{
				/* do worke here but don't modify tp */
			}
			var = tp; // Value does not change after write above
		}
	}
}
