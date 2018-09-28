class T {
	public:
		int val;
		T (int);
		T (const T&);
};

T::T (int v) {
	val = v;
}

T::T(const T& t) {
	val = t.val
}

void g(T a, T b) {
	a.val += b.val;
}


int x = 1;
T a(x);
const T b_aux(x); /* Capture value of x = 1 */
T b(b_aux);
#pragma omp threadprivate(a, b)

void f(int n) {
	x++;
#pragma omp parallel for
	/* In each thread:
 	 * a is constructed from x (with value 1 or 2?)
	 * b is copy-constructred form b_aux
	 */

	for (int i = 0 ;i < n; i++) {
		g(a, b) /* vluaeo f a is unspecified. */
	}
}
