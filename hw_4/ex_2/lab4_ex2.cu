#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

#define NUMOFCUDASTREAMS 4

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int seg_size, int offset) {
    // @@ Insert code to implement vector addition here
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seg_size) return; 

    out[idx + offset] = in1[idx + offset] + in2[idx + offset];
}

// @@ Insert code to implement timer start
// @@ Insert code to implement timer stop
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

    // @@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);


    printf("The input length is %d.\n", inputLength);

    // @@ Insert code below to allocate Host memory for input and output
    cudaHostAlloc(&hostInput1, inputLength * sizeof(DataType), cudaHostAllocDefault);
    cudaHostAlloc(&hostInput2, inputLength * sizeof(DataType), cudaHostAllocDefault);
    cudaHostAlloc(&hostOutput, inputLength * sizeof(DataType), cudaHostAllocDefault);
    cudaHostAlloc(&resultRef, inputLength * sizeof(DataType), cudaHostAllocDefault);

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
    // CUDA streams init
    const int s_Seg = inputLength / NUMOFCUDASTREAMS; // other sizes for profiling: '20', '50', '200', '500', '1000', '10000', '100000'
    const int numSegs = (int) ceil((float) inputLength / s_Seg);

    const int segBytes = s_Seg * sizeof(DataType);
    cudaStream_t stream[NUMOFCUDASTREAMS];
    for (int i = 0; i < NUMOFCUDASTREAMS; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // @@ Initialize the 1D grid and block dimensions here
    int block1D = 128;
    int grid1D = (inputLength + block1D - 1) / block1D;

    double startTime = cpuTiming();

    // asynchronosu copy
    for (int i = 0; i < numSegs; i++) {
        const int offset = i * s_Seg;
        if (i < numSegs - 1) {

            // copy to device
            cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], segBytes, cudaMemcpyHostToDevice, stream[i % NUMOFCUDASTREAMS]);
            cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], segBytes, cudaMemcpyHostToDevice, stream[i % NUMOFCUDASTREAMS]);


            //@@ Launch the GPU Kernel here
            vecAdd<<<grid1D, block1D, 0, stream[i % NUMOFCUDASTREAMS]>>>(deviceInput1, deviceInput2, deviceOutput, s_Seg, offset);

            // back to host (output)
            cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], segBytes, cudaMemcpyDeviceToHost, stream[i % NUMOFCUDASTREAMS]);


        } else {
            // problems occured for input length not a multiple of the number of segments
            const int lastSegSize = inputLength - (numSegs - 1) * s_Seg;
            const int lastSegBytes = lastSegSize * sizeof(DataType);


            // copy to device
            cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], lastSegBytes, cudaMemcpyHostToDevice, stream[i % NUMOFCUDASTREAMS]);
            cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], lastSegBytes, cudaMemcpyHostToDevice, stream[i % NUMOFCUDASTREAMS]);
            
            
            //@@ Launch the GPU Kernel here
            vecAdd<<<grid1D, block1D, 0, stream[i % NUMOFCUDASTREAMS]>>>(deviceInput1, deviceInput2, deviceOutput, lastSegSize, offset);
            
            // back to host (output)
            cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], lastSegBytes, cudaMemcpyDeviceToHost, stream[i % NUMOFCUDASTREAMS]);
        }
    }


    // destroy all initialized cuda streams
    for (int i = 0; i < NUMOFCUDASTREAMS; i++) {
        cudaStreamDestroy(stream[i]);
    }

    cudaDeviceSynchronize();


    double durationTime = cpuTiming() - startTime;
    printf("CUDA Kernel duration of execution time and memory operations (in ms): %f.\n", durationTime);

    //@@ Insert code below to compare the output with the reference
    bool equal = 1;
    for (int i = 0; i < inputLength; i++) {

        //compare if elemetns are equal based on a threshold
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

    // @@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    // @@ Free the CPU memory here
    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    cudaFreeHost(resultRef);

    return 0;
}