class T {
	public:
		static int i;
#pragma omp threadprivate(i)
};
