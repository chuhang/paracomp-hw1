#include <nmmintrin.h>

/*
 * Dimensions for a "kernel" multiply.  We use define statements in
 * order to make sure these are treated as compile-time constants
 * (which the optimizer likes)
 */
#define M 8
#define N 8
#define P 64

/*
 * The ktimer driver expects these variables to be set to whatever
 * the dimensions of a kernel multiply are.  It uses them both for
 * space allocation and for flop rate computations.
 */
int DIM_M=M;
int DIM_N=N;
int DIM_P=P;

#ifdef USE_SHUFPD
#  define swap_sse_doubles(a) _mm_shuffle_pd(a, a, 1)
#else
#  define swap_sse_doubles(a) (__m128d) _mm_shuffle_epi32((__m128i) a, 0x4e)
#endif

/*
 * Block matrix multiply kernel.
 * Inputs:
 *    A: 2-by-P matrix in column major format.
 *    B: P-by-2 matrix in row major format.
 * Outputs:
 *    C: 2-by-2 matrix with element order [c11, c22, c12, c21]
 *       (diagonals stored first, then off-diagonals)
 */
void kdgemm2P2(double * restrict C,
               const double * restrict A,
               const double * restrict B)
{
    // This is really implicit in using the aligned ops...
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);

    // Load diagonal and off-diagonals
    __m128d cd = _mm_load_pd(C+0);
    __m128d co = _mm_load_pd(C+2);

    /*
     * Do block dot product.  Each iteration adds the result of a two-by-two
     * matrix multiply into the accumulated 2-by-2 product matrix, which is
     * stored in the registers cd (diagonal part) and co (off-diagonal part).
     */
    for (int k = 0; k < P; k += 2) {

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

    // Write back sum
    _mm_store_pd(C+0, cd);
    _mm_store_pd(C+2, co);
}

/*
 * Block matrix multiply kernel (simple fixed-size case).
 * Use restrict to tell the compiler there is no aliasing,
 * and inform the compiler of alignment constraints.
 */
void kdgemm(const double * restrict A,
            const double * restrict B,
            double * restrict C)
{
    __assume_aligned(A, 16);
    __assume_aligned(B, 16);
    __assume_aligned(C, 16);
	
	//original code
    /*for (int j = 0; j < N; ++j) {
        for (int k = 0; k < P; ++k) {
            double bkj = B[k+j*P];
            for (int i = 0; i < M; ++i) {
                C[i+j*M] += A[i+k*M]*bkj;
            }
        }
    }*/	
	//code when M N P are all 4
	/*matvec2by2(A,B,C);
	matvec2by2(A,B+2,C+2);*/
	
	for (int i = 0; i < (M/2); i++) 
	{
		for (int j = 0; j < (N/2); j++) 
		{
            kdgemm2P2(C+4*j+i*2*N,A+2*P*i,B+2*P*j);
        }
    }
	/*double* bnew=malloc(2*P*sizeof(double));
	__assume_aligned(bnew,16);
	bnew[0]=B[0];
	bnew[1]=B[2];
	bnew[2]=B[1];
	bnew[3]=B[3];
	kdgemm2P2(C,A,bnew);*/
}

/*
 * Conversion routines that take a matrix block in column-major form
 * and put it into whatever form the kdgemm routine likes.
 */

void to_kdgemm_A(int ldA, const double* restrict A, double * restrict Ak)
{
	__assume_aligned(A, 16);
	__assume_aligned(Ak, 16);
	for (int i = 0; i < (M/2); i++)
	{
       for (int j = 0; j < P; j++)
	   {
			Ak[2*j+i*2*P]=A[2*i+j*M];
			Ak[2*j+1+i*2*P]=A[2*i+1+j*M];
	   }
	}
	/*for (int j = 0; j < N; ++j)
       for (int i = 0; i < M; ++i)
           Ak[i+j*M] = A[i+j*ldA];*/
}

void to_kdgemm_B(int ldB, const double* restrict B, double * restrict Bk)
{
    __assume_aligned(B, 16);
	__assume_aligned(Bk, 16);
	for (int i = 0; i < (N/2); i++)
	{
       for (int j = 0; j < P; j++)
	   {
			Bk[2*j+i*2*P]=B[j+2*P*i];
			Bk[2*j+1+i*2*P]=B[j+(2*i+1)*P];
	   }
	}
	/*for (int j = 0; j < N; ++j)
       for (int i = 0; i < M; ++i)
           Bk[i+j*P] = B[i+j*ldB];*/
}

void from_kdgemm_C(int ldC, const double* restrict Ck, double * restrict C)
{
    __assume_aligned(C, 16);
	__assume_aligned(Ck, 16);
	for (int i = 0; i < (M/2); i++) 
	{
		for (int j = 0; j < (N/2); j++) 
		{
			C[i*2+j*2*M]=Ck[i*2*N+4*j];
			C[i*2+1+(j*2+1)*M]=Ck[i*2*N+4*j+1];
			C[i*2+(j*2+1)*M]=Ck[i*2*N+4*j+2];
			C[i*2+1+j*2*M]=Ck[i*2*N+4*j+3];
		}
	}
	/*for (int j = 0; j < N; ++j)
       for (int i = 0; i < M; ++i)
           C[i+j*ldC] = Ck[i+j*M];*/
}
