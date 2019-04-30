#include <iostream>
#include <xmmintrin.h>			// for Windows; recent version of Clang is OK
#include <cmath>
//#include "assert.hpp"
#include <cassert>
//#include <x86intrin.h>		// for GCC & CLANG

// Example for Using SIMD SSE Intrinsics and Vectorization for regular code;

// Regular Square_Root Function
void SqrtArray_ref(float* __restrict r, const float* __restrict a, int count)		//__restrict__ is a compiler extension C++ pointer_safety option
{																					// default is pointer_safety::relaxed All pointers are considered valid and may be dereferenced or deallocated
																					//__restrict__ ; any pointer pointing to this pointer cannot change the value of variable while this pointer is in scope
	for (int i{ 0 }; i < count; ++i)
	{
		if (a[i] >= 0.0f)
			r[i] = std::sqrt(a[i]);
		else
			r[i] = 0.0f;
	}
}
void SqrtArray_sse(float* __restrict r, const float* __restrict a, int count)
{
	assert(count % 4 == 0);				// assert in x64 config shows error before runtime
	__m128 vz = _mm_set1_ps(0.0f);		// mask with all zeros
	
	for (int i{ 0 }; i < count; i+=4)
	{
		__m128 va = _mm_load_ps(a + i);

		__m128 vq = _mm_sqrt_ps(va);			// if there is any negative value, it will store QaN
		
		// create a mask using the return value of comparision intrinsic 
		__m128 mask = _mm_cmpge_ps(va, vz);		// _mm_cmpge_ps checks va against vz(all zeros) and 
	
		// implement (vq & mask) | (vz & ~mask) // this can be implement as a seperate select function
		__m128 qmask = _mm_and_ps(mask, vq); 
		__m128 znotmask = _mm_andnot_ps(mask, vz);
		__m128 vr = _mm_or_ps(qmask, znotmask);	// select either qmask or znotmask

		
		_mm_store_ps(r+i, vr);				    // store results vr into pointer r (pointing to an empty array of for square root of values)
	}

}


int main()
{
	alignas(16) float values[16] = { 0.0f,1.0f,4.0f,9.0f,
								-16.0f,25.0f,36.0f,49.0f,
								64.0f,81.0f,100.0f,121.0f,
								144.0f,169.0f,196.0f,225.0f };

	constexpr int count{ 16 };

	alignas(16) float r[16];

	SqrtArray_sse(&r[0], &values[0], count);

	std::cout << "\r:[";
	for (const auto elem : r)
		std::cout << elem << " ";
	std::cout << "]\n";

	std::cout << '\n';

	return 0;
}
