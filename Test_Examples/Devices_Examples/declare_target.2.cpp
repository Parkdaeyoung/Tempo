struct typeX
{
	int a;
};

class typeY
{
	int a;
	public:
	int foo() { return a^0x01; }
};

#pragma omp declare target
struct typeX varX;
class typeY varY;
#pragma omp end declare target
void foo()
{
#pragma omp target
	{
		varX.a = 100; // ok
		varY.foo(); // error foo() is not available on a target device
	}
}
