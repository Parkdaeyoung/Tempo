int main()
{
#pragma omp target 
#pragma omp teams distribute
	for (int i = 0; i < 10; i++) 
		i+=1;
	if (1)
		i += 1;
	else
#pragma omp parallel for
		for (int j = 0; j < 10; j++)
			j += j;

}
