#include <iostream>
#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>
#include "dnn.hpp"

//Define the parameters if not defined externally
#ifndef Nn
  #define Nn 128  // Number of Output Layers
  #define Ni 224  // Number of Input  Layers
#endif

#ifndef Tii
  // Tiling Sizes
  #define Tnn 32  
  #define Tii 32
  #define Tn 16
  #define Ti 16
#endif

#define BATCH 16

VTYPE (*synapse)[BATCH][Nn][Ni];
VTYPE (*neuron_i)[BATCH][Ni];
VTYPE (*neuron_n)[BATCH][Nn];
VTYPE (*neuron_n2)[BATCH][Nn];
VTYPE (*neuron_n3)[BATCH][Nn];

void classifier(const VTYPE synapse[Nn][Ni],
                const VTYPE neuron_i[Ni],
                VTYPE neuron_n[Nn]) {
  for (int n = 0; n < Nn; n++) {
    VTYPE temp = 0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = transfer(temp);
  }
}

void classifier_tiled(const VTYPE synapse[Nn][Ni],
                      const VTYPE neuron_i[Ni],
                      VTYPE neuron_n[Nn]) {
  VTYPE sum[Nn] = {};
  for (int outer_n = 0; outer_n < Nn; outer_n += Tnn) { // tiling for output neurons;
    for (int outer_i = 0; outer_i < Ni; outer_i += Tii) { // tiling for input neurons;
      for (int inner_n = outer_n; inner_n < outer_n + Tnn; inner_n += Tn) {
        for (int inner_i = outer_i; inner_i < outer_i + Tii; inner_i += Ti) {
          // Original code
          for (int n = inner_n; n < inner_n + Tn; n++) {
            VTYPE sum_sc = 0;
            for (int i = inner_i; i < inner_i + Ti; i++) {
              sum_sc += synapse[n][i] * neuron_i[i];
            }
            sum[n] += sum_sc;
          }
        }
      }
    }
    for (int n = outer_n; n < outer_n + Tnn; n++) {
      neuron_n[n] = transfer(sum[n]);
    }
  }
}

__global__ void GPU_classifier(const VTYPE synapse[Nn][Ni],
                               const VTYPE neuron_i[Ni],
                               VTYPE neuron_n[Nn],
                               size_t pitch) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  VTYPE *synapse_row = (VTYPE *)((char *)synapse + n * pitch);
  if (n < Nn) {
    VTYPE sum = 0;
    for (int i = 0; i < Ni; i++) {
        sum += synapse_row[i] * neuron_i[i];
    }
    neuron_n[n] = GPU_transfer(sum);
  }
}

__global__ void GPU_classifier_tiled(const VTYPE synapse[Nn][Ni],
                                     const VTYPE neuron_i[Ni],
                                     VTYPE neuron_n[Nn],
                                     size_t pitch,
                                     int tiling_size) {
  VTYPE sum = 0;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  VTYPE *synapse_row = (VTYPE *)((char *)synapse + row * pitch);
  for (int t = 0; t < Ni; t += tiling_size) {
    for (int i = t; i < t + tiling_size; ++i) {
      sum += synapse_row[i] * neuron_i[i];
    }
  }
   neuron_n[row] = GPU_transfer(sum);
}

__global__ void GPU_classifier_tiled_smem(const VTYPE synapse[Nn][Ni],
                                          const VTYPE neuron_i[Ni],
                                          VTYPE neuron_n[Nn],
                                          size_t pitch,
                                          int tiling_size) {
  extern __shared__ VTYPE p[];
  VTYPE *local_synapse = p;
  VTYPE *local_neuron_i = p + Tn * tiling_size;
  VTYPE sum = 0;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  VTYPE *synapse_row = (VTYPE *)((char *)synapse + row * pitch);
  for (int t = 0; t < Ni; t += tiling_size) {
    int col = t * blockDim.x + threadIdx.x;

    local_synapse[threadIdx.y * tiling_size + threadIdx.x] = synapse_row[col];
    local_neuron_i[threadIdx.x] = neuron_i[col];

    __syncthreads();
    for (int i = t; i < t + tiling_size; ++i) {
      sum += local_synapse[threadIdx.y * tiling_size + i] * local_neuron_i[i];
    }
   __syncthreads();
  }
   neuron_n[row] = GPU_transfer(sum);
}

int main(void) {
  std::cout << "------ Initializing ------" << std::endl;

  synapse   = (VTYPE (*)[BATCH][Nn][Ni]) aligned_alloc(64, BATCH * Nn * Ni * sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[BATCH][Ni]) aligned_alloc(64, BATCH * Ni * sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[BATCH][Nn]) aligned_alloc(64, BATCH * Nn * sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[BATCH][Nn]) aligned_alloc(64, BATCH * Nn * sizeof(VTYPE));
  neuron_n3 = (VTYPE (*)[BATCH][Nn]) aligned_alloc(64, BATCH * Nn * sizeof(VTYPE));

  fill_random((VTYPE *) synapse, Nn * Ni * BATCH);
  fill_random((VTYPE *) neuron_i, Ni * BATCH);
  memset(neuron_n, 0, Nn * BATCH* sizeof(VTYPE));
  memset(neuron_n2, 0, Nn * BATCH * sizeof(VTYPE));
  memset(neuron_n3, 0, Nn * BATCH * sizeof(VTYPE));

  std::cout << "------ Running CPU version ------" << std::endl;
  std::cout << "Simple version: \t";

  timeit([]() {
    for(int b = 0; b < BATCH; ++b) {
      classifier((*synapse)[b], (*neuron_i)[b], (*neuron_n)[b]);
    }
  });

  std::cout << "Tiled version:  \t";  
  timeit([]() {
    for(int b = 0; b < BATCH; ++b) {
      classifier_tiled((*synapse)[b], (*neuron_i)[b], (*neuron_n2)[b]);
    }
  });

  compare((VTYPE *)neuron_n, (VTYPE *)neuron_n2, Nn * BATCH);

  std::cout << "------ Running GPU version ------" << std::endl;
  VTYPE (*d_synapse)[Ni];
  VTYPE *d_neuron_i, *d_neuron_n;

  size_t pitch;
  cudaMallocPitch((void **)&d_synapse, &pitch, Ni * sizeof(VTYPE), Nn);
  cudaMalloc(&d_neuron_i, Ni * sizeof(VTYPE));
  cudaMalloc(&d_neuron_n, Nn * sizeof(VTYPE));

  cudaMemcpy2D(d_synapse, pitch, synapse, Ni * sizeof(VTYPE), Ni * sizeof(VTYPE), Nn, cudaMemcpyHostToDevice);
  cudaMemcpy(d_neuron_i, neuron_i, Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);

  std::cout << "Simple version: \t";


  int num_threads = 8;
  int num_blocks = Nn / num_threads;

  cudaMemset(d_neuron_n, 0, Nn * sizeof(VTYPE));
  CUDA_timeit([&]() {
    GPU_classifier<<<num_blocks, num_threads>>>(d_synapse, d_neuron_i, d_neuron_n, pitch);
  });
  cudaMemcpy(neuron_n3, d_neuron_n, Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);

  compare(neuron_n, neuron_n3, Nn);
  std::cout << "Tiled version:\t";

/*  for(int ti = 2; ti <= 512; ti *= 2) {
  cudaMemset(d_neuron_n, 0, Nn * sizeof(VTYPE));
  dim3 blockDim(1, Tn), gridDim(1, Nn/Tn);
  CUDA_timeit([&]() {
    GPU_classifier_tiled<<<gridDim, blockDim>>>(d_synapse, d_neuron_i, d_neuron_n, pitch, ti);
  });
  auto err = cudaGetLastError();
  if(err) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  cudaMemcpy(neuron_n2, d_neuron_n, Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
  std::cout << "tiled size = " << ti << std::endl;

  compare(neuron_n, neuron_n2, Nn);
  }*/
}
