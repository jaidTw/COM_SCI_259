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

VTYPE synapse[Nn][Ni] __attribute__((aligned(64)));
VTYPE neuron_i[Ni] __attribute__((aligned(64)));
VTYPE neuron_n[Nn] __attribute__((aligned(64)));
VTYPE neuron_n2[Nn] __attribute__((aligned(64)));
VTYPE neuron_n3[Nn] __attribute__((aligned(64)));

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
                               const VTYPE *neuron_i,
                               VTYPE *neuron_n) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < Nn) {
    VTYPE sum = 0;
    for (int i = 0; i < Ni; i++) {
        sum += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = GPU_transfer(sum);
  }
}
/*
__global__ void GPU_classifier_layer_blocked_compute(const VTYPE synapse[Tn][Ti],
                                                     const VTYPE *neuron_i,
                                                     VTYPE *neuron_n) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= Tn)
    return;
  VTYPE sum = 0;
  for (int i = 0; i < Ti; i++) {
      sum += synapse[n][i] * neuron_i[i];
  }
  neuron_n[n] = GPU_transfer(sum);
}

void GPU_classifier_layer_blocked(VTYPE synapse[Nn][Ni],
                                  VTYPE neuron_i[Ni], 
                                  VTYPE neuron_n[Nn]) {
  VTYPE sum[Nn] = {};
  for (int outer_n = 0; outer_n < Nn; outer_n += Tnn) { // tiling for output neurons;

    for (int outer_i = 0; outer_i < Ni; outer_i += Tii) { // tiling for input neurons;
      for (int inner_n = outer_n; inner_n < outer_n + Tnn; inner_n += Tn) {
        for (int inner_i = outer_i; inner_i < outer_i + Tii; inner_i += Ti) {
//          GPU_classifier_layer_blocked_compute<<<4, Tn/4>>>();
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
*/
int main(void) {
  std::cout << "------ Initializing ------" << std::endl;

  fill_random((VTYPE *) synapse, Nn * Ni);
  fill_random((VTYPE *) neuron_i, Ni);
  memset(neuron_n, 0, Nn * sizeof(VTYPE));
  memset(neuron_n2, 0, Nn * sizeof(VTYPE));
  memset(neuron_n3, 0, Nn * sizeof(VTYPE));

  std::cout << "------ Running CPU version ------" << std::endl;
  std::cout << "Simple version: \t";
  auto f1 = std::bind(classifier, synapse, neuron_i, neuron_n);
  timeit(f1);

  std::cout << "Blocked version:\t";  
  auto f2 = std::bind(classifier_tiled, synapse, neuron_i, neuron_n2);
  timeit(f2);

  compare(neuron_n, neuron_n2, Nn);

  std::cout << "------ Running GPU version ------" << std::endl;
  VTYPE (*d_synapse)[Ni];
  VTYPE *d_neuron_i, *d_neuron_n;

  cudaMalloc((void **)&d_synapse, Nn * Ni * sizeof(VTYPE));
  cudaMalloc(&d_neuron_i, Ni * sizeof(VTYPE));
  cudaMalloc(&d_neuron_n, Nn * sizeof(VTYPE));

  cudaMemcpy(d_synapse, synapse, Nn * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(d_neuron_i, neuron_i, Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);

  std::cout << "Simple version: \t";

  for (int num_threads = 2; num_threads <= 1024; num_threads *= 2) {
    int num_blocks = Nn / num_threads;

    cudaMemset(d_neuron_n, 0, Nn * sizeof(VTYPE));
    timeit([&]() {
      GPU_classifier<<<num_blocks, num_threads>>>(d_synapse, d_neuron_i, d_neuron_n);
    });
    cudaMemcpy(neuron_n3, d_neuron_n, Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);

    std::cout << "#threads = " << num_threads << ", #blocks = " << num_blocks << std::endl;
    compare(neuron_n, neuron_n3, Nn);
  }
  std::cout << "Blocked version:\t";

  GPU_classifier_layer_blocked(d_synapse, d_neuron_i, d_neuron_n);
}
