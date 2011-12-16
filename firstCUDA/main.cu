/*
* CURAND API: inizio uso dei PseudoRandom Number Generator
* limitazione a 65535 kernel, lancio monodimensionale/bidimensionale
* attenzione al numero N...non riesco a processare molti dati usando in contemporanea il monitor
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

#define PI 3.14159265358979323846
#define N 128
#define N2 N

__global__ void setup_kernel ( curandStateXORWOW_t * state){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x+ y*blockDim.x*gridDim.x;
	while (offset<N){
/* Each thread gets same seed , a different sequence number no offset */
		curand_init (1234 , offset , 0 , &state[offset]);
		offset += blockDim.x*gridDim.x;
		__syncthreads();
	}
	
}

__global__ void generate_bit_kernel ( curandStateXORWOW_t * state , float * result ){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;
	while (offset < N){
		curandStateXORWOW_t localState = state[offset];
		float awgn = curand_normal(&localState);
		result[offset] = awgn;
		//state[offset]=localState;
		offset += blockDim.x*gridDim.x;
		__syncthreads();
	}
	
}

int main ( int argc , char * argv []){
	int i;
	//dim3 dimGrid(1024); //numero block
    //dim3 dimBlock(512); //numero threads per block
	dim3 dimGrid(8,8); 
    dim3 dimBlock(16,16); // 2 dim max 512
	curandStateXORWOW_t * devStates ;
	float *hostResults, *devResults;
	/* Allocate space for results on host */
	hostResults = ( float *) calloc (N2 , sizeof(float) );
	/* Allocate space for results on device */
	cudaMalloc (( void **) &devResults , N2 * sizeof(float) );
	/* Set results to 0 */
	cudaMemset ( devResults , 2, N2 * sizeof(float) );
	/* Allocate space for prng states on device */
	cudaMalloc (( void **) &devStates , N2 * sizeof(curandStateXORWOW_t) );
	/* Setup prng states */
	setup_kernel <<<dimGrid, dimBlock>>>( devStates ) ;
	cudaThreadSynchronize();
	/* Generate and use pseudo - random */
	//for ( i = 0; i < 10; i++) {
		generate_bit_kernel <<<dimGrid,dimBlock>>>( devStates , devResults ) ;
		cudaThreadSynchronize();
	//}
	/* Copy device memory to host */
	cudaMemcpy ( hostResults , devResults , N2 * sizeof(float) , cudaMemcpyDeviceToHost ) ;
	/* Show result */
	float tmp;
	for ( i=0; i < N; i++ ) {
		tmp+=hostResults[i];
	}
	printf("%f \n",tmp);
	/* Cleanup */
	cudaFree(devStates);
	cudaFree(devResults);
	free(hostResults);
	system("pause");
	return EXIT_SUCCESS;
}