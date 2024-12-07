#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <omp.h>
#include <omp.h>
#include <climits> 
#include <math.h>
#include <vector>
using namespace std;

#define bs 256
#define SEED 123
#define DIGITS 10

// Cada thread toma el digito de un elemento del array A con respecto al exponente dado y aumenta
// un contador de ocurrencias de dicho digito
__global__ void histogram_kernel(int n, int *dA, int *dHist, int exp) {
    // Histograma con ocurrencias de cada digito en memoria compartida
    __shared__ int sharedHist[DIGITS];
    
    // Se limpia el histograma de resultados anteriores
    if (threadIdx.x < DIGITS) sharedHist[threadIdx.x] = 0;
    __syncthreads();

    // Se obtiene digito de elemento de A y se usa atomicAdd para añadir ocurrencia a sharedHist
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < n) {
        int digit = (dA[tidx] / exp) % DIGITS;
        atomicAdd(&sharedHist[digit], 1);
    }
    __syncthreads();

    // Se guarda resultados de memoria compartida en el histograma dHist
    if (threadIdx.x < DIGITS) {
        atomicAdd(&dHist[threadIdx.x], sharedHist[threadIdx.x]);
    }
}

// 
__global__ void scatter_kernel(int n, int *dA, int *dR, int *dPrefixSum, int exp) {
    // prefixSum en memoria compartida
    __shared__ int sharedPrefix[DIGITS];

    // se carga la suma de prefijos a la memoria compartida para mejorar rendimiento 
    if (threadIdx.x < DIGITS) {
        sharedPrefix[threadIdx.x] = dPrefixSum[threadIdx.x];
    }
    __syncthreads();

    // Se obtiene digito del elemento de A al que accede el thread
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < n) {
        int digit = (dA[tidx] / exp) % DIGITS;

        // Se obtiene posicion global que ocupara el elemento de A y se guarda en R
        int pos = atomicAdd(&sharedPrefix[digit], 1);
        dR[pos] = dA[tidx];
    }
}

// Realiza prefixSum con el histograma y versiones anteriores del prefixSum y guarda en el mismo prefix
void exclusive_scan(int *hist, int *prefix, int size) {
    prefix[0] = 0;
    for (int i = 1; i < size; ++i) {
        prefix[i] = prefix[i - 1] + hist[i - 1];
    }
}

// PSKC Radix-sort
void cpu(int *&data, int n, int b);

void gpu(int *A, int *R, int *dA, int *dR, int n);

// llena array de tamaño n con enteros
void llena_array(int *A, int n, int nt, int seed);

// Imprime un array de enteros
void print_array(int n, int *array);

// Ejecutar como ./prog n modo nt b
int main(int argc, char **argv){

    // Tomar argumentos e inicializar variables
    int n = atoi(argv[1]);
    int modo = atoi(argv[2]); 
    int nt = atoi(argv[3]);
    int b = atoi(argv[4]);
    omp_set_num_threads(nt);

    int *A = new int[n];
    int *R = new int[n]; // arreglos en memoria principal
    int *dA;
    int *dR; // direcciones de arreglos en gpu

    // Llenar array e imprimir si es lo bastante pequeño
    llena_array(A, n, nt, SEED);
    if(n <= 32){
        cout << "Array de entrada:" << endl;
        print_array(n, A);
    }

    if (modo == 0){
        double inicio = omp_get_wtime();
        cpu(R, n, b);
        double fin = omp_get_wtime();
        cout << "Tiempo de ejecución: " << fin - inicio << " segundos" << endl;
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

void llena_array(int *A, int n, int nt, int seed){
    #pragma omp parallel shared(A)
    {
        int tid = omp_get_thread_num();
        long chunk = n / nt;
        long start = tid * chunk;
        long end = (tid == nt - 1) ? n : start + chunk;

        std::mt19937 mt(seed + tid);
        std::uniform_int_distribution<int> dist(0, n - 1);

        for (int k = start; k < end; ++k) {
            A[k] = dist(mt);
        }
    }
}

void print_array(int n, int *array){
    for(int i = 0; i < n; ++i){
        printf("%d ", array[i]);
    }
    cout << endl;
}

void cpu(int *&data, int n, int b){
    cout << "Comenzando..en funcion" << endl;
    // Paso 1: Obtener muestra de datos
    // Sample: contiene elementos seleccionados a intervalos regulares del vector data, con el objetivo no tener que procesar todo el vector original
    int* sample = new int[n];
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        sample[i] = data[i * (n / b)];
    }
    cout << "listo paso 1" << endl;
    // Paso 2: Reverse Sorting local
    // El Reverse Sorting local se refiere a ordenar un subconjunto (sample) del vector data en orden descendente (de mayor a menor).
    sort(sample, sample + n); // ordena el array sample de manera ascendente (de menor a mayor)
    
    // A partir de los valores ordenados en sample, se seleccionan b valores que servirán como particiones. Estos valores se almacenan en el arreglo partitions.

    vector<int> partitions(b + 1); // b particiones
    // se están eligiendo b elementos equidistantes del vector sample ordenado, que servirán como valores de partición.
    for (int i = 0; i < b; i++) {
        partitions[i] = sample[i * (n / b)]; 
    }
    partitions[b] = INT_MAX; // para tener un valor de partición que sea mayor que cualquier elemento del vector data.
    cout << "listo paso 2" << endl;
    // Paso 3: Comprobar equilibrio de carga y elección
    
    // Se averigua si se podria conseguir un buen equilibrio de carga entre los procesadores tras dos iteraciones de Reverse Sorting. 

    int total_work = 0;
    #pragma omp parallel for reduction(+:total_work)
    for (int i = 0; i < n; i++) {
        int bucket = std::lower_bound(partitions.begin(), partitions.end(), data[i]) - partitions.begin() - 1;
        total_work += bucket;
    }
    
    double imbalance = static_cast<double>(total_work) / (n * b);
    cout << "listo paso 3" << endl;
    // Si aplicando Reverse Sorting NO se consiguiese un buen equilibrio de carga entre los datos muestreados, se estara en el Caso (3.2) 
    if (imbalance > 1.1) {
        // Caso 3.1: Aplicar Reverse Sorting paralelo y ordenar.
        // Todos los procesadores particionan el conjunto de datos a ordenar con la tecnica de Reverse Sorting paralelo.
        cout << "en caso 3.1" << endl;
        #pragma omp parallel for
        for (int i = 0; i < b; i++) {
            sort(data + i * (n / b), data + (i + 1) * (n / b), greater<int>());
        }
    } else {
        // Caso 3.2: Aplicar Counting Split paralelo y ordenar.
        // Los procesadores particionan conCounting Split en paralelo. 
        cout << "en caso 3.2" << endl;
        vector<vector<int>> buckets(b);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            int bucket_index = lower_bound(partitions.begin(), partitions.end(), data[i]) - partitions.begin() - 1;
            #pragma omp critical
            {
                buckets[bucket_index].push_back(data[i]);
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < b; i++) {
            sort(buckets[i].begin(), buckets[i].end());
        }

        int index = 0;
        for (const auto& bucket : buckets) {
            for (int num : bucket) {
                data[index++] = num;
            }
        }
    }

    // Limpiar memoria de particiones y muestra
    delete[] sample;
}


void gpu(int *A, int *R, int *dA, int *dR, int n) {
    // Reservar memoria en gpu para histograma y prefixsum
    int *dHist, *dPrefixSum;
    cudaMalloc(&dHist, DIGITS * sizeof(int));
    cudaMalloc(&dPrefixSum, DIGITS * sizeof(int));

    // Dimensiones de la grilla
    dim3 blockSize(bs, 1, 1);
    dim3 gridSize((n + bs - 1) / bs, 1, 1);

    // Inicializar histograma y prefixsum en memoria principal
    int *hHist = new int[DIGITS];
    int *hPrefixSum = new int[DIGITS];

    // Para medir tiempo en milisegundos
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Comienza a medir tiempo
    cudaEventRecord(start);

    // A partir del numero mas grande del array se sabe cuantos digitos usar
    int maxVal = *max_element(A, A + n);

    // El for se ejecuta por cada digito, es decir unidades, decenas, etc
    for (int exp = 1; maxVal / exp > 0; exp *= 10) {
        // Se resetea el histograma de gpu
        cudaMemset(dHist, 0, DIGITS * sizeof(int));

        // Se crea el histograma para los digitos actuales
        histogram_kernel<<<gridSize, blockSize>>>(n, dA, dHist, exp);
        cudaDeviceSynchronize();

        // Se copia el histograma de gpu al de memoria principal
        cudaMemcpy(hHist, dHist, DIGITS * sizeof(int), cudaMemcpyDeviceToHost);

        // Se calcula prefixSum
        exclusive_scan(hHist, hPrefixSum, DIGITS);

        // Se copia prefixSum en memoria principal
        cudaMemcpy(dPrefixSum, hPrefixSum, DIGITS * sizeof(int), cudaMemcpyHostToDevice);

        // Se ordenan elementos de acuerdo al digito actual con el prefixSum 
        scatter_kernel<<<gridSize, blockSize>>>(n, dA, dR, dPrefixSum, exp);
        cudaDeviceSynchronize();

        // Se intercambian punteros de A y R para trabajar sobre el arreglo parcialmente ordenado
        // con respecto al ultimo digito
        std::swap(dA, dR);
    }

    // Se termina de contar el tiempo
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Se guarda el resultado en memoria principal
    cudaMemcpy(R, dA, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Se imprime tiempo que tomo ordenar
    std::cout << "Time taken by GPU radix sort: " << milliseconds << " ms" << std::endl;

    // Liberar memoria
    cudaFree(dHist);
    cudaFree(dPrefixSum);
    delete[] hHist;
    delete[] hPrefixSum;

    // Se eliminan eventos de cuda
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

