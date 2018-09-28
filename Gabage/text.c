void vecAdd(double *A, double *B, double *C, int N) {
#pragma omp target map(to:A[0:N], B[0:N]) map(from:C[0:N]) map(to:N)

	{
		//struct (anonymous at vecAdd_b.c:2:9) *const restrict __context

		// Initial Memory Management
			int created_A;
		double * m_A = CreateOrGetBuffer(&A[0], N * sizeof(A[0]), 1, &created_A);
		int created_B;
		double * m_B = CreateOrGetBuffer(&B[0], N * sizeof(B[0]), 1, &created_B);
		int created_C;
		double * m_C = CreateOrGetBuffer(&C[0], N * sizeof(C[0]), 0, &created_C);
		int created_N;
		int m_N = CreateOrGetBuffer(&N, sizeof(N), 1, created_N);


		// Memory cleanup
		DestroyBuffer(&A[0], N * sizeof(A[0]), 1, created_A);
		DestroyBuffer(&A, sizeof(A), 1, created_A);
		DestroyBuffer(&B[0], N * sizeof(B[0]), 1, created_B);
		DestroyBuffer(&B, sizeof(B), 1, created_B);
		DestroyBuffer(&C[0], N * sizeof(C[0]), 0, created_C);
		DestroyBuffer(&C, sizeof(C), 0, created_C);
		DestroyBuffer(&N, sizeof(N), 1, created_N);

	}
	{
#pragma omp teams num_teams(32) thread_limit(64)

		{
			//const int *const restrict .global_tid.
			//const int *const restrict .bound_tid.
			//struct (anonymous at vecAdd_b.c:4:9) *const restrict __context


			// team configuration
			PushNumTeams(32);
			PushThreadLimit(64);

			{
#pragma omp distribute parallel for
				for (int i = 0; i < N; ++i) {
					C[i] = A[i] + B[i];
				}

				// team configuration clean up
				PopNumTeams(32);
				PopThreadLimit(64);

			}
		}
	}
}

