#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

__device__ DataType add(DataType a, DataType b) {
    return a + b;
}

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    //@@ Insert code to implement vector addition here
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;

    out[idx] = add(in1[idx], in2[idx]);
}

//@@ Insert code to implement timer start
//@@ Insert code to implement timer stop
double cpuTiming() {
   struct timeval elapsedTime;
   gettimeofday(&elapsedTime,NULL);
   
   return ((double)elapsedTime.tv_sec*1.e3 + (double)elapsedTime.tv_usec*1.e-3);
}

int main(int argc, char **argv) {
    int inputLength;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    //@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);


    printf("The input length is %d.\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output
    hostInput1 = (DataType*) malloc(inputLength * sizeof(DataType));
    hostInput2 = (DataType*) malloc(inputLength * sizeof(DataType));
    hostOutput = (DataType*) malloc(inputLength * sizeof(DataType));
    resultRef = (DataType*) malloc(inputLength * sizeof(DataType));

    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    for (int i = 0; i < inputLength; i++) {
        DataType rndNum1 = rand() / (DataType) RAND_MAX; // Random number in interval [0, 1.0]
        DataType rndNum2 = rand() / (DataType) RAND_MAX; // Random number in interval [0, 1.0]


        hostInput1[i] = rndNum1;
        hostInput2[i] = rndNum2;
        resultRef[i] = rndNum1 + rndNum2;
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));



    //@@ Insert code to below to Copy memory to the GPU here
    double startTime = cpuTiming();

    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    double durationTime = cpuTiming() - startTime;
    printf("Data copy (H2D) execution time (in ms): %f.\n", durationTime);

    //@@ Initialize the 1D grid and block dimensions here
    int block1D = 128;
    int grid1D = (inputLength + block1D - 1) / block1D;

    //@@ Launch the GPU Kernel here
    startTime = cpuTiming();

    vecAdd <<<grid1D, block1D>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    cudaDeviceSynchronize();

    durationTime = cpuTiming() - startTime;
    printf("CUDA Kernel duration of execution time (in ms): %f.\n", durationTime);

    //@@ Copy the GPU memory back to the CPU here
    startTime = cpuTiming();

    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    durationTime = cpuTiming() - startTime;
    printf("Data copy (D2H) execution time (in ms): %f.\n", durationTime);

    //@@ Insert code below to compare the output with the reference
    bool equal = 1;
    for (int i = 0; i < inputLength; i++) {


        //comparison of elements based on an error threshold
        if (fabs(hostOutput[i] - resultRef[i]) > 1e-7) { 
            equal = 0;
            break;
        }
    }
    if (equal) {
        printf("CPU & GPU results are equal.\n");
    } else {
        printf("CPU & GPU results are NOT equal.\n");
    }

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return 0;
}
