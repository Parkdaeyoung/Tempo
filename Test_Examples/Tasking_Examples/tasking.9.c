void work()
{
#pragma omp task
	{ // Task 1
#pragma omp task
		{ // Task 2
#pragma omp critical // Critical region 1
			{ /* do work here */ }
		}
#pragma omp critical // Ciritical region 2
		{
			// Capture data for the following task
#pragma omp task
			{ /* do work here */ }
		}
	}
}
