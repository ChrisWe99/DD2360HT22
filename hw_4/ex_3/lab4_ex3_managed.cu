#include <stdio.h>
#include <sys/time.h>

#define DataType float 

//#define DataType double 

// matrix multiplication for CPU
void matrixMultiplication(DataType *A, DataType *B, DataType *C, int numARows,
            int numACols, int numBRows, int numBCols) {
    // Input matrices: A, B. Output: C
    for (int i = 0; i < numARows; i++) {
      
        for (int j = 0; j < numBCols; j++) {
            C[i*numBCols + j] = 0.0;

            for (int k = 0; k < numACols; k++) {
                C[i*numBCols + j] += A[i*numACols + k] * B[k*numBCols + j];
            }
        }
    }
}

// matrix multiplication for GPU
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows, int numACols, int numBRows, int numBCols){
    //@@ Insert code to implement matrix multiplication here
    
    const int rows = blockIdx.y * blockDim.y + threadIdx.y;
    const int cols = blockIdx.x * blockDim.x + threadIdx.x;


    if ((cols >= numBCols) || (rows >= numARows)){
      return;
    } 

    // temp result before addition of the single multiplications
    DataType tmpResult = 0.0;

    // calculate final result
    for (int k = 0; k < numACols; k++) {
        tmpResult += A[rows*numACols + k] * B[k*numBCols + cols];
    }
    C[rows*numBCols + cols] = tmpResult;
}

int main(int argc, char **argv) {
    DataType *matrixA; // The A matrix
    DataType *matrixB; // The B matrix
    DataType *matrixC; // The output C matrix
    DataType *resultRef; // The reference result


    int numARows;    // number of rows in the matrix A
    int numACols; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBCols; // number of columns in the matrix B
    int numCRows;
    int numCCols;

    //@@ Insert code below to read in numARows, numACols, numBCols from args

    numARows = atoi(argv[1]);
    numACols = atoi(argv[2]);
    numBCols = atoi(argv[3]);

    numBRows = numACols;
    numCRows = numARows;
    numCCols = numBCols;


    printf("Dimensions of input matrix A (%d x %d), B (%d x %d), and output matrix C (%d x %d)\n", numARows, numACols, numBRows, numBCols, numCRows, numCCols);


    //@@ Insert code below to allocate Host memory for input and output
    // allocate memory for sizeof datatype for each element of both input matrices and the output matrices
    /*hostA = (DataType*) malloc(numARows * numACols * sizeof(DataType)); 
    hostB = (DataType*) malloc(numBRows * numBCols * sizeof(DataType)); 
    hostC = (DataType*) malloc(numCRows * numCCols * sizeof(DataType)); */

    //pinned mem alloc
    /*cudaHostAlloc(&hostA, numARows * numACols * sizeof(DataType), cudaHostAllocDefault); 
    cudaHostAlloc(&hostB, numBRows * numBCols * sizeof(DataType), cudaHostAllocDefault); 
    cudaHostAlloc(&hostC, numCRows * numCCols * sizeof(DataType), cudaHostAllocDefault); 
    resultRef = (DataType*) malloc(numCRows * numCCols * sizeof(DataType)); */

    //managed mem alloc
    cudaMallocManaged(&matrixA, numARows * numCCols * sizeof(DataType)); 
    cudaMallocManaged(&matrixB, numBRows * numCCols * sizeof(DataType)); 
    cudaMallocManaged(&matrixC, numCRows * numCCols * sizeof(DataType)); 

    cudaMallocManaged(&resultRef, numCRows * numCCols * sizeof(DataType)); //result

    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numACols; j++) {
          // random number from 0 to 1
            DataType rndNum = rand() / (DataType) RAND_MAX;
            matrixA[i*numACols + j] = rndNum;
        }
    }
    for (int i = 0; i < numBRows; i++) {
        for (int j = 0; j < numBCols; j++) {
          // random number from 0 to 1
            DataType rndNum = rand() / (DataType) RAND_MAX;
            matrixB[i*numBCols + j] = rndNum;
        }
    }
    // Calculate reference result
    matrixMultiplication(matrixA, matrixB, resultRef, numARows, numACols, numBRows, numBCols);

    //not needed here
    /*
    //@@ Insert code below to allocate GPU memory here
    // allocate memory for sizeof datatype for each element of both input matrices and the output matrices
    cudaMalloc(&deviceA, numARows * numACols * sizeof(DataType)); 
    cudaMalloc(&deviceB, numBRows * numBCols * sizeof(DataType)); 
    cudaMalloc(&deviceC, numCRows * numCCols * sizeof(DataType)); 

    //@@ Insert code to below to Copy memory to the GPU here
    // copy memory for sizeof datatype for each element of both input matrices
    cudaMemcpy(deviceA, hostA, numARows * numACols * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBCols * sizeof(DataType), cudaMemcpyHostToDevice); */

    //@@ Initialize the grid and block dimensions here
    
    int DbCols = 16;
    int DbRows = 16;
    int DgCols = (numCCols + DbCols - 1) / DbCols;
    int DgRows = (numCRows + DbRows - 1) / DbRows;

    //@@ Launch the GPU Kernel here
    gemm <<<dim3(DgCols, DgRows, 1), dim3(DbCols, DbRows, 1)>>>(matrixA, matrixB, matrixC, numARows, numACols, numBRows, numBCols);

    /*
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCRows * numCCols * sizeof(DataType), cudaMemcpyDeviceToHost); */

    //syncing was also suggested by different online posts
    cudaDeviceSynchronize();

    //@@ Insert code below to compare the output with the reference
    bool equal = 1;
    for (int i = 0; i < numCRows; i++) {
        for (int j = 0; j < numCCols; j++) {
          // compare if single elements are equal based on an error threshold
            if (fabs(matrixC[i*numCCols + j] - resultRef[i*numCCols + j]) > 1e-3) {
                equal = 0;
                break;
            }
        }
    }


    
    if (equal) {
        printf("CPU & GPU results are equal.\n");
    } else {
        printf("CPU & GPU results are NOT equal.\n");
    }

    /*
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC); */

    //@@ Free the CPU memory here
    cudaFree(matrixA);
    cudaFree(matrixB);
    cudaFree(matrixC);

    cudaFree(resultRef);

    return 0;
}