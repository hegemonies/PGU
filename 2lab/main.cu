#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
	 fprintf(stderr, "Error %s at line %d in file %s\n",\
	 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
	 exit(1);\
	} }

double w_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1E-6;
}

__global__ void init(float *a, float *b)
{
	a[threadIdx.x + blockDim.x * blockIdx.x] = threadIdx.x + blockDim.x * blockIdx.x;
	b[threadIdx.x + blockDim.x * blockIdx.x] = threadIdx.x + blockDim.x * blockIdx.x;
}

__global__ void compute(float *a, float *b, float *c)
{
	c[threadIdx.x + blockDim.x * blockIdx.x] = a[threadIdx.x + blockDim.x * blockIdx.x] + b[threadIdx.x + blockDim.x * blockIdx.x];
}

int main()
{
	int blocks = 781;
	int th_p_block = 128;
	int N = blocks * th_p_block;
	float *a_device;
	float *b_device;
	float *c_device;
	float *buffer_host;
	float elapsedTime;
	int start = pow(2, 10);
	int end = pow(2, 15); // in the task 2^23 
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	while (start < end) {
		printf("start = %d\n", start);
		blocks = start;
		th_p_block = 1;
		N = blocks * th_p_block;
		
		while (th_p_block <= 32) {
			// printf("N = %d\tblocks = %d\tth_p_block = %d\n", N, blocks, th_p_block);
	
			// elapsedTime = 0;
	
			buffer_host = (float *)malloc(N * sizeof(float));
	
			CUDA_CHECK_RETURN(cudaMalloc(&a_device, N * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc(&b_device, N * sizeof(float)));
			CUDA_CHECK_RETURN(cudaMalloc(&c_device, N * sizeof(float)));
	
			init<<<blocks, th_p_block>>>(a_device, b_device);
	
			// elapsedTime -= w_time();
			cudaEventRecord(startEvent, 0);

			compute<<<blocks, th_p_block>>>(a_device, b_device, c_device);

			cudaEventRecord(stopEvent, 0);
			cudaEventSynchronize(stopEvent);

			cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

			// CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			CUDA_CHECK_RETURN(cudaGetLastError()); 
			// elapsedTime += w_time();
	
			printf("%f\n", elapsedTime);
	
			CUDA_CHECK_RETURN(cudaMemcpy(buffer_host, c_device, N * sizeof(float), cudaMemcpyDeviceToHost));
	
			free(buffer_host);
			cudaFree(a_device);
			cudaFree(b_device);
			cudaFree(c_device);
			
			blocks /= 2;
			th_p_block *= 2;
			N = blocks * th_p_block;
		}

		start += 32;

		printf("\n");
	}
		
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	return 0;
}