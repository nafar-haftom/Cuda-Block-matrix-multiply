//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY are used to set the number of threads in a CUDA block 
#define TILEX 16
#define TILEY 8

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILEX,n/TILEY);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY);
	return dimBlock;
}
__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {

    int row = by * TILEY + ty;
    int col = bx * TILEX + tx;

    __shared__ float As[TILEY][TILEX*4];
    __shared__ float Bs[TILEY*4][TILEX];

    float Csub = 0.0;

    for (int p = 0 ; p < (n/TILEY) ; p = p + 4){
		
		//loop unrolling
		if(tx < TILEY){
			As[ty][tx] = ad[((row)<<(m)) + (TILEY*p+tx)];
		}
		if(tx < TILEY){
			As[ty][tx + TILEY] = ad[((row)<<(m)) + (TILEY*(p+1)+tx)];
		}
		if(tx < TILEY){
			As[ty][tx + 2*TILEY] = ad[((row)<<(m)) + (TILEY*(p+2)+tx)];
		}
		if(tx < TILEY){
			As[ty][tx + 3*TILEY] = ad[((row)<<(m)) + (TILEY*(p+3)+tx)];
		}
		//loop unrolling
		if (ty < TILEX){
			Bs[ty][tx] = bd[((TILEY*p + ty)<<(m)) + (col)];
		}
		if (ty < TILEX){
			Bs[ty + TILEY][tx] = bd[((TILEY * (p+1) + ty) << (m)) + (col)];
		}
		if (ty < TILEX){
			Bs[ty + 2*TILEY][tx] = bd[((TILEY * (p+2) + ty) << (m)) + (col)];
		}
		if (ty < TILEX){
			Bs[ty + 3*TILEY][tx] = bd[((TILEY * (p+3) + ty) << (m)) + (col)];
		}
		__syncthreads();
		
		for (int k = 0 ; k < 4 * TILEY ; k++){ 
			Csub += As[ ty ][ k ] * Bs[ k ][ tx ];
		}
		__syncthreads();
	}
	
	cd [( (row) << (m) ) + ( col )] = Csub;
}
