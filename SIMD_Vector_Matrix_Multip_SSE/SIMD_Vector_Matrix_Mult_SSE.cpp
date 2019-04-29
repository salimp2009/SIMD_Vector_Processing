#include <iostream>
#include <xmmintrin.h>
//#include <x86intrin.h>   // for GCC and Clang

// Example for Multiplying four-element vector with a 4 x 4 matrix
// using SIMD Vectors and SSE Intrinsics

// Using union to access members of matrix as floats
union Mat44
{
	float c[4][4];  //components; 4 rows with 4 elements
	__m128 row[4]; // 4 SSE vectors to represent 4 rows
};

__m128 MulVecMat_sse(const __m128& v, const Mat44& M)
{
	//first transpose v using shuffle masks
	__m128 vX = _mm_shuffle_ps(v, v, 0x00); // (vx, vx, vx, vx)
	__m128 vY = _mm_shuffle_ps(v, v, 0x55); // (vy, vy, vy, vy)
	__m128 vZ = _mm_shuffle_ps(v, v, 0xAA); // (vz, vz, vz, vz)
	__m128 vW = _mm_shuffle_ps(v, v, 0xFF); // (vz, vz, vz, vz)

	__m128	r =				  _mm_mul_ps(vX, M.row[0]);		// multiply vX and Mx (M.row[0])
			r = _mm_add_ps(r, _mm_mul_ps(vY, M.row[1]));	// multiply vY and My (M.row[1]) and add to r
			r = _mm_add_ps(r, _mm_mul_ps(vZ, M.row[2]));	// multiply vZ and Mz (M.row[2]) and add to r
			r = _mm_add_ps(r, _mm_mul_ps(vW, M.row[3]));	// multiply vW and Mw (M.row[3]) and add to r

	return r;
}

// Matrix-Matrix Multilication using SSE
void MulMatMat_sse(Mat44& R, const Mat44& A, const Mat44& B)
{
	R.row[0] = MulVecMat_sse(A.row[0], B);
	R.row[1] = MulVecMat_sse(A.row[1], B);
	R.row[2] = MulVecMat_sse(A.row[2], B);
	R.row[3] = MulVecMat_sse(A.row[3], B);
}

int main()
{
	alignas(16) float a[16] = { 0.0f,1.0f,2.0f,3.0f,
						4.0f,5.0f,6.0f,7.0f,
						8.0f,9.0f,10.0f,11.0f,
						12.0f,13.0f,14.0f,15.0f };

	alignas(16) float b[16] = { 0.0f,1.0f,2.0f,3.0f,
							4.0f,5.0f,6.0f,7.0f,
							8.0f,9.0f,10.0f,11.0f,
							12.0f,13.0f,14.0f,15.0f };
	
	
	Mat44 R;
	Mat44 A;
	Mat44 B;

	A.row[0]=  _mm_load_ps(&a[0]);
	A.row[1] = _mm_load_ps(&a[4]);
	A.row[2] = _mm_load_ps(&a[8]);
	A.row[3] = _mm_load_ps(&a[12]);

	B.row[0] = _mm_load_ps(&b[0]);
	B.row[1] = _mm_load_ps(&b[4]);
	B.row[2] = _mm_load_ps(&b[8]);
	B.row[3] = _mm_load_ps(&b[12]);

	MulMatMat_sse(R, A, B);

	for (int i{ 0 }; i < 4; ++i) {
		std::cout << "row: [";
		for (int j{ 0 }; j < 4; ++j) {
			std::cout << R.c[i][j] << " ";
		}
		std::cout << "]\n";
	}
	std::cout << '\n';


	return 0;
}