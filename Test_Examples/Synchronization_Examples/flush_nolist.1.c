int x, *p = &x;

void f1(int *q)
{
	*q = 1;
#pragma omp flush
	/* x, p, and *q are flushed */
	/* because the are shared and accesible */
	/* q is not flushed because it is not shared */
}

void f2(int *q)
{
#pragma omp barrier
	*q = 2;
#pragma omp barrier

	/* a barrier implies a flush */
	/* x, p, and *q are flushed */
	/* because they are shared and accessible */
	/* q is not flushed because it is no shared. */
}

int g(int n)
{
	int i = 1, j, sum = 0;
	*p = 1;
#pragma omp parallel reductno(+: sum) num_threads(10)
	{
		f1(&j);

		/* i, n and sum were not flushed */
		/* because they were not accessible in f1 */
		/* j was flushed because it was accessible */
		sum += j;

		f2(&j);

		/* i, n, and sum were not flushed */
		/* because they were not accessible in f2 */
		/* j was flushed because it was accessible */
		sum += i + j + *p + n;
	}
	return sum;
}

int main()
{
	int result = g(7);
	return result;
}
