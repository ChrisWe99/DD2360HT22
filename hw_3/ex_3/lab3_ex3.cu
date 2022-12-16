#include <stdio.h>
#include <sys/time.h>
#include <random>

//define for the Max value of 127 according to the given assignment
#define MAX_VALUE 127
#define NUM_BINS 4096


void histogram_cpu(unsigned int *input, unsigned int *bins,
                   unsigned int num_elements, unsigned int num_bins) {

    // Initialize all array elements with 0
    memset(bins, 0, num_bins * sizeof(*bins));

    // Count frequency of the different bin values inside the elements
    for (int i = 0; i < num_elements; ++i) {

        unsigned int value = input[i];
        //if condition necessary to avoid exceeding the max value of 127 as specified in the assignment
        if (bins[value] < MAX_VALUE) {
            bins[value] += 1;
        }
    }
}

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements, unsigned int num_bins) {
    //@@ Insert code below to compute histogram of input using shared memory and atomics
    //https://forums.developer.nvidia.com/t/difference-between-threadidx-blockidx-statements/12161
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared variable for each threadblock
    __shared__ unsigned int shared_bins[NUM_BINS];

    // Initialize the the shared bin frequency values to 0
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        // if condition necessary because there may be more threads than num_bins
        if (i < num_bins) {
            shared_bins[i] = 0;
        }
    }
    // thread syncing (otherwise I got errors sometimes)
    __syncthreads();


    // execute an atomic addition for all elements of the block
    if (idx < num_elements) {
        atomicAdd(&(shared_bins[input[idx]]), 1);
    }


    // thread syncing (otherwise I got errors sometimes)
    __syncthreads();


    // Comobine all shared histograms to the final one
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        // if condition necessary because there may be more threads than num_bins
        if (i < num_bins) {
            atomicAdd(&(bins[i]), shared_bins[i]);
        }
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    //@@ Insert code below to clean up bins that saturate at MAX_VALUE
    //https://forums.developer.nvidia.com/t/difference-between-threadidx-blockidx-statements/12161
    const int bin = blockIdx.x * blockDim.x + threadIdx.x;

    if (bin >= num_bins){
        return;
    }

    else if (bins[bin] > MAX_VALUE) {
        bins[bin] = MAX_VALUE;
    }
}

int main(int argc, char **argv) {
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    //@@ Insert code below to read in inputLength from args
    inputLength = atoi(argv[1]);

    printf("Chosen input problem length: %d\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output
    hostBins = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));

    hostInput = (unsigned int*) malloc(inputLength * sizeof(unsigned int));

    resultRef = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));

    //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    for (int i = 0; i < inputLength; i++) {
        //formula from here: https://www.geeksforgeeks.org/generating-random-number-range-c/
        // num = (rand() % (upper - lower + 1)) + lower;
        hostInput[i] = rand() % NUM_BINS;  
    }

    //@@ Insert code below to create reference result in CPU
    histogram_cpu(hostInput, resultRef, inputLength, NUM_BINS);

    //@@ Insert code below to allocate GPU memory here

    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    

    //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    //@@ Initialize the grid and block dimensions here
    int Db_hist = 64;

    int Dg_hist = (inputLength + Db_hist - 1) / Db_hist;

    //@@ Launch the GPU Kernel here
    histogram_kernel<<<Dg_hist, Db_hist>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

    //@@ Initialize the second grid and block dimensions here
    int Db_convert = 64;

    int Dg_convert = (NUM_BINS + Db_convert - 1) / Db_convert;

    //@@ Launch the second GPU Kernel here
    convert_kernel<<<Dg_convert, Db_convert>>>(deviceBins, NUM_BINS);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    bool equal = 1;
    for (int i = 0; i < NUM_BINS; i++) {

        if (hostBins[i] != resultRef[i]) {
            equal = 0;
        }
    }
    if (equal) {
        printf("CPU & GPU results are equal.\n");
    } else {
        printf("CPU & GPU results are NOT equal.\n");
    }

    // save generated histogram in a file
    FILE *fptr;

    fptr = fopen("./histogram.txt","w+");

    for (int i = 0; i < NUM_BINS; i++) {
        fprintf(fptr, "%d\n", hostBins[i]);
    }

    fclose(fptr);

    //@@ Free the GPU memory here
    cudaFree(deviceBins);

    cudaFree(deviceInput);
    

    //@@ Free the CPU memory here
    free(hostBins);

    free(hostInput);
    
    free(resultRef);

    return 0;
}