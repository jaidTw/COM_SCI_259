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

VTYPE (*synapse)[Nn][Ni];
VTYPE (*neuron_i)[Ni];
VTYPE (*neuron_n)[Nn];
VTYPE (*neuron_n2)[Nn];
VTYPE (*neuron_n3)[Nn];

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

__global__ void GPU_classifier(const cudaPitchedPtr synapse,
                               const cudaPitchedPtr neuron_i,
                               const cudaPitchedPtr neuron_n,
                               size_t batch_begin,
                               int mn) {
  /* x => n, y => batch */
  int b = blockIdx.y + batch_begin;
  int n_begin = (blockIdx.x * blockDim.x + threadIdx.x) * mn;

  VTYPE * const neuron_i_row = (VTYPE *)((char *)neuron_i.ptr + b * neuron_i.pitch);
  VTYPE * const neuron_n_row = (VTYPE *)((char *)neuron_n.ptr + b * neuron_n.pitch);
  for (int n = n_begin; n < n_begin + mn; n++) {
    VTYPE sum = 0;
    VTYPE * const synapse_row = (VTYPE *)((char *)synapse.ptr + (b * Nn + n) * synapse.pitch);
    for (int i = 0; i < Ni; i++) {
        sum += synapse_row[i] * neuron_i_row[i];
    }
    neuron_n_row[n] = GPU_transfer(sum);
  }
}

__global__ void GPU_classifier_tiled(const cudaPitchedPtr synapse,
                                     const cudaPitchedPtr neuron_i,
                                     const cudaPitchedPtr neuron_n,
                                     size_t batch_begin,
                                     int mn, int tiling_size) {
  /* x => n, y => batch */
  int b = blockIdx.y + batch_begin;
  int n_begin = (blockIdx.x * blockDim.x + threadIdx.x) * mn;

  VTYPE * const neuron_i_row = (VTYPE *)((char *)neuron_i.ptr + b * neuron_i.pitch);
  VTYPE * const neuron_n_row = (VTYPE *)((char *)neuron_n.ptr + b * neuron_n.pitch);
  for (int n = n_begin; n < n_begin + mn; n++) {
    VTYPE sum = 0;
    VTYPE * const synapse_row = (VTYPE *)((char *)synapse.ptr + (b * Nn + n) * synapse.pitch);
    for (int i = 0; i < Ni; i += tiling_size) {
      for(int ii = i; ii < i + tiling_size; ii++) {
        sum += synapse_row[ii] * neuron_i_row[ii];
      }
    }
    neuron_n_row[n] = GPU_transfer(sum);
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage : " << argv[0] << " BATCH_SIZE BATCH_IN_PARALLEL" << std::endl;
    exit(0);
  }
  const int batch = strtol(argv[1], nullptr, 10);
  const int batch_in_parallel = strtol(argv[2], nullptr, 10);
  if (batch_in_parallel > batch) {
    std::cerr << "BATCH_IN_PARALLEL must smaller than BATCH" << std::endl;
    exit(0);
  } else if (batch % batch_in_parallel) {
    std::cerr << "BATCH must be a multiple of BATCH_IN_PARALLEL" << std::endl;
    exit(0);
  }

  std::cout << "Initializing ..." << std::endl;

  synapse   = (VTYPE (*)[Nn][Ni]) aligned_alloc(64, batch * Nn * Ni * sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[Ni]) aligned_alloc(64, batch * Ni * sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[Nn]) aligned_alloc(64, batch * Nn * sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[Nn]) aligned_alloc(64, batch * Nn * sizeof(VTYPE));
  neuron_n3 = (VTYPE (*)[Nn]) aligned_alloc(64, batch * Nn * sizeof(VTYPE));

  fill_random((VTYPE *) synapse, Nn * Ni * batch);
  fill_random((VTYPE *) neuron_i, Ni * batch);
  memset(neuron_n, 0, Nn * batch * sizeof(VTYPE));
  memset(neuron_n2, 0, Nn * batch * sizeof(VTYPE));
  memset(neuron_n3, 0, Nn * batch * sizeof(VTYPE));

  std::cout << "CPU Simple version: \t\t\t\t\t";
  timeit([&]() {
    for(int b = 0; b < batch; ++b) {
      classifier(synapse[b], neuron_i[b], neuron_n[b]);
    }
  });

  std::cout << "CPU Tiled version:  \t\t\t\t\t";
  timeit([&]() {
    for(int b = 0; b < batch; ++b) {
      classifier_tiled(synapse[b], neuron_i[b], neuron_n2[b]);
    }
  });

  compare((VTYPE *)neuron_n, (VTYPE *)neuron_n2, Nn * batch);

  cudaExtent extent_synapse = make_cudaExtent(Ni * sizeof(VTYPE), Nn, batch);
  cudaPitchedPtr d_synapse;
  MallocAndCpy3D(d_synapse, synapse, extent_synapse);

  size_t pitch_neuron_i, pitch_neuron_n;
  VTYPE (*_d_neuron_i)[Ni], (*_d_neuron_n)[Nn];
  cudaMallocPitch((void **)&_d_neuron_i, &pitch_neuron_i, Ni * sizeof(VTYPE), batch);
  cudaMallocPitch((void **)&_d_neuron_n, &pitch_neuron_n, Nn * sizeof(VTYPE), batch);
  cudaPitchedPtr d_neuron_i = make_cudaPitchedPtr(_d_neuron_i, pitch_neuron_i, Ni, batch);
  cudaPitchedPtr d_neuron_n = make_cudaPitchedPtr(_d_neuron_n, pitch_neuron_n, Nn, batch);
  cudaMemcpy2D(d_neuron_i.ptr, d_neuron_i.pitch, neuron_i, Ni * sizeof(VTYPE), Ni * sizeof(VTYPE), batch, cudaMemcpyHostToDevice);

  std::cout << "GPU version:\n";

  int mn = 1;
  int threads_total = Nn / mn;
  for(int block_num = 1; block_num < threads_total; block_num *= 2) {
    int block_num = 64;
    dim3 block_size(threads_total / block_num);
    dim3 grid_size(block_num, batch_in_parallel);
    if (threads_total / block_num > 1024 || threads_total/block_num == 0)
      continue;

    cudaMemset(d_neuron_n.ptr, 0, batch * Nn * sizeof(VTYPE));

    printf("Grid: (%4d, %4d), Block: (%4d, %4d), Mn=%4d\t", grid_size.x, grid_size.y, block_size.x, block_size.y, mn);
    CUDA_timeit([&]() {
      for(int b = 0; b < batch; b += batch_in_parallel) {
        GPU_classifier<<<grid_size, block_size>>>(d_synapse, d_neuron_i, d_neuron_n, b, mn);
      }
    });
    cudaMemcpy2D(neuron_n3, Nn * sizeof(VTYPE), d_neuron_n.ptr, d_neuron_n.pitch, Nn * sizeof(VTYPE), batch, cudaMemcpyDeviceToHost);

    cuda_check_error();
    compare((VTYPE *)neuron_n, (VTYPE *)neuron_n3, Nn * batch);
  }
}
