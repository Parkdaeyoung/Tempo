float dotprod(float B[], float C[], int N)
{
	float sum = 0;
	int i;
	//#pragma omp target teams map(to: B[0:N], C[0:N])\
	//	defaultmap(tofrom:scalar) reduction(+:sum)
	{
		//const int *const restrict .global_tid.
		//const int *const restrict .bound_tid.
		//struct (anonymous at Test_Examples/Devices_Examples/teams.3.c:5:9) *const restrict __context


		// team configuration
		PushNumTeams();
		PushThreadLimit();


		{
			//const int *const restrict .global_tid.
			//const int *const restrict .bound_tid.
			//struct (anonymous at Test_Examples/Devices_Examples/teams.3.c:5:9) *const restrict __context


			// Initial Memory Management
			int created_B;
			float * m_B = CreateOrGetBuffer(&B[0], N * sizeof(B[0]), 1, &created_B);
			int created_C;
			float * m_C = CreateOrGetBuffer(&C[0], N * sizeof(C[0]), 1, &created_C);


#pragma omp distribute parallel for reduction(+:sum)
			for (i = 0; i < N; i++)
				sum += B[i] * C[i];
			// team configuration clean up
			PopNumTeams();
			PopThreadLimit();

		}

		// Memory cleanup
		DestroyBuffer(&B[0], N * sizeof(B[0]), 1, created_B);
		DestroyBuffer(&C[0], N * sizeof(C[0]), 1, created_C);
	}

	return sum;
}
/* Note: The variables sum0, sum1 are now mapped with tofrom, for correct
 * execution with 4.5 (and pre-4.5) compliant compilers.
 * See Devices Intro.
 */
