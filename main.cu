#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
using namespace std;

#define bs 256


#define DIGITS 10

__global__ void histogram_kernel(int n, int *dA, int *dHist, int exp) {
    __shared__ int sharedHist[DIGITS];
    
    // Initialize shared histogram
    if (threadIdx.x < DIGITS) sharedHist[threadIdx.x] = 0;
    __syncthreads();

    // Compute digit for each element
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < n) {
        int digit = (dA[tidx] / exp) % DIGITS;
        atomicAdd(&sharedHist[digit], 1);
    }
    __syncthreads();

    // Write shared histogram to global memory
    if (threadIdx.x < DIGITS) {
        atomicAdd(&dHist[threadIdx.x], sharedHist[threadIdx.x]);
    }
}

__global__ void scatter_kernel(int n, int *dA, int *dR, int *dPrefixSum, int exp) {
    __shared__ int sharedPrefix[DIGITS];

    // Load prefix sums into shared memory
    if (threadIdx.x < DIGITS) {
        sharedPrefix[threadIdx.x] = dPrefixSum[threadIdx.x];
    }
    __syncthreads();

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < n) {
        int digit = (dA[tidx] / exp) % DIGITS;

        // Compute global position and scatter element
        int pos = atomicAdd(&sharedPrefix[digit], 1);
        dR[pos] = dA[tidx];
    }
}

void exclusive_scan(int *hist, int *prefix, int size) {
    prefix[0] = 0;
    for (int i = 1; i < size; ++i) {
        prefix[i] = prefix[i - 1] + hist[i - 1];
    }
}


void cpu();

void gpu(int *A, int *R, int *dA, int *dR, int n);

// llena array de tamaño n con enteros
void llena_array(int n, int *array);

// Imprime un array de enteros
void print_array(int n, int *array);

// Ejecutar como ./prog n modo nt
int main(int argc, char **argv){
    srand(static_cast<unsigned int>(time(0)));

    // Tomar argumentos e inicializar variables
    int n = atoi(argv[1]);
    int modo = atoi(argv[2]); 
    int nt = atoi(argv[3]);

    int *A = new int[n];
    int *R = new int[n]; // arreglos en memoria principal
    int *dA;
    int *dR; // direcciones de arreglos en gpu
    
    // Llenar array e imprimir si es lo bastante pequeño
    llena_array(n, A);
    if(n <= 32){
        cout << "Array de entrada:" << endl;
        print_array(n, A);
    }

    if (modo == 0){
        cpu();
    }
    else{
        // allocar memoria en device  (GPU)
        cudaMalloc(&dA, n * sizeof(int));
        cudaMalloc(&dR, n * sizeof(int));

        // copiar de Host -> Device
        cudaMemcpy(dA, A, sizeof(int)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(dR, R, sizeof(int)*n, cudaMemcpyHostToDevice);

        gpu(A, R, dA, dR, n);


    }

    // Imprimir resultado si es lo bastante pequeño
    if(n<= 32){
        cout << "Array de salida:" << endl;
        print_array(n, R);
    }

    // Liberar memoria
    delete[] A;
    delete[] R;

}

void llena_array(int n, int *array) {
    for (int i = 0; i < n; ++i) {
        array[i] = rand() % 1000;
    }
}

void print_array(int n, int *array){
    for(int i = 0; i < n; ++i){
        printf("%d ", array[i]);
    }
    cout << endl;
}

void cpu(){
    return;
}

void gpu(int *A, int *R, int *dA, int *dR, int n) {
    // Allocate memory for histogram and prefix sum
    int *dHist, *dPrefixSum;
    cudaMalloc(&dHist, DIGITS * sizeof(int));
    cudaMalloc(&dPrefixSum, DIGITS * sizeof(int));

    dim3 blockSize(bs, 1, 1);
    dim3 gridSize((n + bs - 1) / bs, 1, 1);

    int *hHist = new int[DIGITS];
    int *hPrefixSum = new int[DIGITS];

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer
    cudaEventRecord(start);

    int maxVal = *max_element(A, A + n);
    for (int exp = 1; maxVal / exp > 0; exp *= 10) {
        // Reset histogram
        cudaMemset(dHist, 0, DIGITS * sizeof(int));

        // Step 1: Compute histogram
        histogram_kernel<<<gridSize, blockSize>>>(n, dA, dHist, exp);
        cudaDeviceSynchronize();

        // Copy histogram to host and compute prefix sum
        cudaMemcpy(hHist, dHist, DIGITS * sizeof(int), cudaMemcpyDeviceToHost);
        exclusive_scan(hHist, hPrefixSum, DIGITS);

        // Copy prefix sum to device
        cudaMemcpy(dPrefixSum, hPrefixSum, DIGITS * sizeof(int), cudaMemcpyHostToDevice);

        // Step 2: Scatter elements
        scatter_kernel<<<gridSize, blockSize>>>(n, dA, dR, dPrefixSum, exp);
        cudaDeviceSynchronize();

        // Swap input and output arrays for next iteration
        std::swap(dA, dR);
    }

    // End the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy final sorted array to host
    cudaMemcpy(R, dA, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the time taken
    std::cout << "Time taken by GPU radix sort: " << milliseconds << " ms" << std::endl;

    // Free memory
    cudaFree(dHist);
    cudaFree(dPrefixSum);
    delete[] hHist;
    delete[] hPrefixSum;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

