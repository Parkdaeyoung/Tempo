int main()
{
	int i[100];
#pragma omp target
#pragma omp teams num_teams(2) thread_limit(2)
	{
		int k;
		{
			int j = k;
			i[0] += j;
		}
	}

}
