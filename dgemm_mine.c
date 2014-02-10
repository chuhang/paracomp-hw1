#include <nmmintrin.h>
#ifdef USE_SHUFPD
#  define swap_sse_doubles(a) _mm_shuffle_pd(a, a, 1)
#else
#  define swap_sse_doubles(a) (__m128d) _mm_shuffle_epi32((__m128i) a, 0x4e)
#endif

#define MAX_SIZE 769u
const char* dgemm_desc = "My awesome dgemm.";

void kdgemm2P2(const int M,double * restrict C,
               const double * restrict A,
               const double * restrict B)
{
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);

    __m128d cd = _mm_load_pd(C+0);
    __m128d co = _mm_load_pd(C+2);

    for (int k = 0; k < M; k += 2) {

        __m128d a0 = _mm_load_pd(A+2*k+0);
        __m128d b0 = _mm_load_pd(B+2*k+0);
        __m128d td0 = _mm_mul_pd(a0, b0);
        __m128d bs0 = swap_sse_doubles(b0);
        __m128d to0 = _mm_mul_pd(a0, bs0);

        __m128d a1 = _mm_load_pd(A+2*k+2);
        __m128d b1 = _mm_load_pd(B+2*k+2);
        __m128d td1 = _mm_mul_pd(a1, b1);
        __m128d bs1 = swap_sse_doubles(b1);
        __m128d to1 = _mm_mul_pd(a1, bs1);

        __m128d td_sum = _mm_add_pd(td0, td1);
        __m128d to_sum = _mm_add_pd(to0, to1);

        cd = _mm_add_pd(cd, td_sum);
        co = _mm_add_pd(co, to_sum);
    }

    _mm_store_pd(C+0, cd);
    _mm_store_pd(C+2, co);
}

/*void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < M; ++j) {
            double cij = C[j*M+i];
            for (k = 0; k < M; ++k)
                cij += A[k*M+i] * B[j*M+k];
            C[j*M+i] = cij;
        }
    }
}*/

void square_dgemm(const int M,const double * restrict A,const double * restrict B,double * restrict C)
{
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);
	
	double* Ak = (double*) malloc(M * M * sizeof(double));
    double* Bk = (double*) malloc(M * M * sizeof(double));
    double* Ck = (double*) malloc(M * M * sizeof(double));
	
	__assume_aligned(Ak, 16);
	__assume_aligned(Bk, 16);
	__assume_aligned(Ck, 16);

	memset(Ck,0,M *M * sizeof(double));
	
	for (int i = 0; i < (M/2); i++)
	{
       for (int j = 0; j < M; j++)
	   {
			Ak[2*j+i*2*M]=A[2*i+j*M];
			Ak[2*j+1+i*2*M]=A[2*i+1+j*M];
	   }
	}
	
	for (int i = 0; i < (M/2); i++)
	{
       for (int j = 0; j < M; j++)
	   {
			Bk[2*j+i*2*M]=B[j+2*M*i];
			Bk[2*j+1+i*2*M]=B[j+(2*i+1)*M];
	   }
	}
	
	for (int i = 0; i < (M/2); i++) 
	{
		for (int j = 0; j < (M/2); j++) 
		{
            kdgemm2P2(M,Ck+4*j+i*2*M,Ak+2*M*i,Bk+2*M*j);
        }
    }
	
	for (int i = 0; i < (M/2); i++) 
	{
		for (int j = 0; j < (M/2); j++) 
		{
			C[i*2+j*2*M]=Ck[i*2*M+4*j];
			C[i*2+1+(j*2+1)*M]=Ck[i*2*M+4*j+1];
			C[i*2+(j*2+1)*M]=Ck[i*2*M+4*j+2];
			C[i*2+1+j*2*M]=Ck[i*2*M+4*j+3];
		}
	}
}

/*void to_kdgemm_A(const int M, const double* restrict A, double * restrict Ak)
{
	__assume_aligned(A, 16);
	__assume_aligned(Ak, 16);
	for (int i = 0; i < (M/2); i++)
	{
       for (int j = 0; j < M; j++)
	   {
			Ak[2*j+i*2*M]=A[2*i+j*M];
			Ak[2*j+1+i*2*M]=A[2*i+1+j*M];
	   }
	}
}

void to_kdgemm_B(const int M, const double* restrict B, double * restrict Bk)
{
    __assume_aligned(B, 16);
	__assume_aligned(Bk, 16);
	for (int i = 0; i < (M/2); i++)
	{
       for (int j = 0; j < M; j++)
	   {
			Bk[2*j+i*2*M]=B[j+2*M*i];
			Bk[2*j+1+i*2*M]=B[j+(2*i+1)*M];
	   }
	}
}

void from_kdgemm_C(const int M, const double* restrict Ck, double * restrict C)
{
    __assume_aligned(C, 16);
	__assume_aligned(Ck, 16);
	for (int i = 0; i < (M/2); i++) 
	{
		for (int j = 0; j < (M/2); j++) 
		{
			C[i*2+j*2*M]=Ck[i*2*M+4*j];
			C[i*2+1+(j*2+1)*M]=Ck[i*2*M+4*j+1];
			C[i*2+(j*2+1)*M]=Ck[i*2*M+4*j+2];
			C[i*2+1+j*2*M]=Ck[i*2*M+4*j+3];
		}
	}
}*/
