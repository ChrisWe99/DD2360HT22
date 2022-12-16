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
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numACols, int numBRows, int numBCols){
    //@@ Insert code to implement matrix multiplication here
    
    const int rows = blockIdx.x * blockDim.x + threadIdx.x;
    const int cols = blockIdx.y * blockDim.y + threadIdx.y;


    if ((rows >= numBCols) || (cols >= numARows)){
      return;
    } 

    // temp result before addition of the single multiplications
    DataType tmpResult = 0.0;
    for (int k = 0; k < numACols; k++) {
        tmpResult += A[cols*numACols + k] * B[k*numBCols + rows];
    }
    C[cols*numBCols + rows] = tmpResult;
}

int main(int argc, char **argv) {
    DataType *hostA; // The A matrix
    DataType *hostB; // The B matrix
    DataType *hostC; // The output C matrix
    DataType *resultRef; // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
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
    hostA = (DataType*) malloc(numARows * numACols * sizeof(DataType)); 
    hostB = (DataType*) malloc(numBRows * numBCols * sizeof(DataType)); 
    hostC = (DataType*) malloc(numCRows * numCCols * sizeof(DataType)); 
    resultRef = (DataType*) malloc(numCRows * numCCols * sizeof(DataType));

    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numACols; j++) {
          // random number from 0 to 1
            DataType rndNum = rand() / (DataType) RAND_MAX;
            hostA[i*numACols + j] = rndNum;
        }
    }
    for (int i = 0; i < numBRows; i++) {
        for (int j = 0; j < numBCols; j++) {
          // random number from 0 to 1
            DataType rndNum = rand() / (DataType) RAND_MAX;
            hostB[i*numBCols + j] = rndNum;
        }
    }
    // Calculate reference result
    matrixMultiplication(hostA, hostB, resultRef, numARows, numACols, numBRows, numBCols);

    //@@ Insert code below to allocate GPU memory here
    // allocate memory for sizeof datatype for each element of both input matrices and the output matrices
    cudaMalloc(&deviceA, numARows * numACols * sizeof(DataType)); 
    cudaMalloc(&deviceB, numBRows * numBCols * sizeof(DataType)); 
    cudaMalloc(&deviceC, numCRows * numCCols * sizeof(DataType)); 

    //@@ Insert code to below to Copy memory to the GPU here
    // copy memory for sizeof datatype for each element of both input matrices
    cudaMemcpy(deviceA, hostA, numARows * numACols * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBCols * sizeof(DataType), cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
    
    int DbCols = 16;
    int DbRows = 16;
    int DgCols = (numCCols + DbCols - 1) / DbCols;
    int DgRows = (numCRows + DbRows - 1) / DbRows;

    //@@ Launch the GPU Kernel here
    gemm <<<dim3(DgCols, DgRows, 1), dim3(DbCols, DbRows, 1)>>>(deviceA, deviceB, deviceC, numARows, numACols, numBRows, numBCols);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCRows * numCCols * sizeof(DataType), cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    bool equal = 1;
    for (int i = 0; i < numCRows; ++i) {
        for (int j = 0; j < numCCols; ++j) {
          // compare if single elements are equal based on an error threshold
            if (fabs(hostC[i*numCCols + j] - resultRef[i*numCCols + j]) > 1e-4) {
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

    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    //@@ Free the CPU memory here
    free(hostA);
    free(hostB);
    free(hostC);
    free(resultRef);

    return 0;
}