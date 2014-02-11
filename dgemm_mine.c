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
	double* Ak = (double*) malloc((M+1) * (M+1) * sizeof(double));
    double* Bk = (double*) malloc((M+1) * (M+1) * sizeof(double));
    double* Ck = (double*) malloc((M+1) * (M+1) * sizeof(double));	
	__assume_aligned(Ak, 16);
	__assume_aligned(Bk, 16);
	__assume_aligned(Ck, 16);
	memset(Ck, 0, (M+1) * (M+1) * sizeof(double));
	
	if ((M%2)==0)
	{	
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
	else
	{
	double* temp = (double*) malloc((M+1) * (M+1) * sizeof(double));	
	__assume_aligned(temp, 16);
	
	for (int i = 0; i < M; i++)
	{
       for (int j = 0; j < M; j++)
	   {
			temp[j+i*(M+1)]=A[j+i*M];
	   }
	   temp[M+i*(M+1)]=0;
	}
	for (int j = 0; j < (M+1); j++)
	{
		temp[M*(M+1)+j]=0;
	}
	
	for (int i = 0; i < ((M+1)/2); i++)
	{
       for (int j = 0; j < (M+1); j++)
	   {
			Ak[2*j+i*2*(M+1)]=temp[2*i+j*(M+1)];
			Ak[2*j+1+i*2*(M+1)]=temp[2*i+1+j*(M+1)];
	   }
	}
	
	/*printf("A matrix:\n");
    for (int i = 0; i < DIM_M; ++i) {
        for (int j = 0; j < DIM_N; ++j)
            printf(" %g", A[i+j*DIM_M]);
        printf("\n");
    }
	printf("Ak:\n");
	for (int i = 0; i < ((DIM_M+1)*(DIM_M+1)); ++i)
		printf(" %g", Ak[i]);
	printf("\n");*/
	
	for (int i = 0; i < M; i++)
	{
       for (int j = 0; j < M; j++)
	   {
			temp[j+i*(M+1)]=B[j+i*M];
	   }
	   temp[M+i*(M+1)]=0;
	}
	for (int j = 0; j < (M+1); j++)
	{
		temp[M*(M+1)+j]=0;
	}
	for (int i = 0; i < ((M+1)/2); i++)
	{
       for (int j = 0; j < (M+1); j++)
	   {
			Bk[2*j+i*2*(M+1)]=temp[j+2*(M+1)*i];
			Bk[2*j+1+i*2*(M+1)]=temp[j+(2*i+1)*(M+1)];
	   }
	}
	/*printf("B matrix:\n");
    for (int i = 0; i < DIM_M; ++i) {
        for (int j = 0; j < DIM_N; ++j)
            printf(" %g", B[i+j*DIM_M]);
        printf("\n");
    }
	printf("Bk:\n");
	for (int i = 0; i < ((DIM_M+1)*(DIM_M+1)); ++i)
		printf(" %g", Bk[i]);
	printf("\n");*/
	
	for (int i = 0; i < ((M+1)/2); i++) 
	{
		for (int j = 0; j < ((M+1)/2); j++) 
		{
            kdgemm2P2((M+1),Ck+4*j+i*2*(M+1),Ak+2*(M+1)*i,Bk+2*(M+1)*j);
        }
    }
	
	for (int i = 0; i < ((M+1)/2); i++) 
	{
		for (int j = 0; j < ((M+1)/2); j++) 
		{
			temp[i*2+j*2*(M+1)]=Ck[i*2*(M+1)+4*j];
			temp[i*2+1+(j*2+1)*(M+1)]=Ck[i*2*(M+1)+4*j+1];
			temp[i*2+(j*2+1)*(M+1)]=Ck[i*2*(M+1)+4*j+2];
			temp[i*2+1+j*2*(M+1)]=Ck[i*2*(M+1)+4*j+3];
		}
	}
	for (int i = 0; i < M; i++)
	{
       for (int j = 0; j < M; j++)
	   {
			C[j+i*M]=temp[j+i*(M+1)];
	   }
	}
	}
}

