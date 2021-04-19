#include <iostream>
#include <string>
#include <functional>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  
  #define Ty  8
  #define Tx  8
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)

VTYPE (*synapse)[Ky][Kx][Nn][Ni];
VTYPE (*neuron_i)[NYPAD][NXPAD][Ni];
VTYPE (*neuron_n)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n2)[NYSCL][NXSCL][Nn];
VTYPE (*neuron_n3)[NYSCL][NXSCL][Nn];

void convolution_tiled(const VTYPE synapse[Ky][Kx][Nn][Ni], 
                       const VTYPE neuron_i[NYPAD][NXPAD][Ni], 
                       VTYPE neuron_n[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn] = {};

  for (int yy = 0; yy < Ny; yy += Ty) {
    for (int xx = 0; xx < Nx; xx += Tx) {
      for (int nnn = 0; nnn < Nn; nnn += Tnn) {
        int yout = yy / Sy;
        for (int y = yy; y < yy + Ty; y += Sy) { // tiling for y;
          int xout = xx / Sx;

          for (int x = xx; x < xx + Tx; x += Sx) { // tiling for x;

            for (int nn = nnn; nn < nnn + Tnn; nn += Tn) {
              memset(sum + nn, 0, Tn * sizeof(VTYPE));

              for (int ky = 0; ky < Ky; ky++) {  // sliding window;
                for (int kx = 0; kx < Kx; kx++) {

                  int ii = 0;
                  VTYPE sum_sc;

                  for (; ii < Ni -Ti+1; ii += Ti) {
                    for (int n = nn; n < nn + Tn; n++) {
                      sum_sc=0;
                      for (int i = ii; i < ii + Ti; i++) {
                        VTYPE sv = synapse[ky][kx][n][i];
                        VTYPE nv = neuron_i[ky + y][kx + x][i];
                        sum_sc += sv * nv;
                      }
                      sum[n] += sum_sc;
                    }
                  }
                }
              }

              //transfer
              for (int n = nn; n < nn + Tn; n++) {
                neuron_n[yout][xout][n] = transfer(sum[n]);
              }
            }
            xout++; 
          }
          yout++;
        }
      }
    }
  }
}

void convolution(const VTYPE synapse[Ky][Kx][Nn][Ni], 
                 const VTYPE neuron_i[NYPAD][NXPAD][Ni], 
                 VTYPE neuron_n[NYSCL][NXSCL][Nn]) {
  VTYPE sum[Nn] = {};

  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Ny; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        memset(sum + nn, 0, Tn * sizeof(VTYPE));

        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
                VTYPE sv = synapse[ky][kx][n][i];
                VTYPE nv = neuron_i[ky + y][kx + x][i];
                sum[n] += sv * nv;
              }
        for (int n = nn; n < nn + Tn; n++) {
          neuron_n[yout][xout][n] = transfer(sum[n]);
        }
      }
      xout++; 
    }
    yout++;
  }
}


__global__ void GPU_convolution(const VTYPE synapse[Ky][Kx][Nn][Ni], 
                                const VTYPE neuron_i[NYPAD][NXPAD][Ni], 
                                VTYPE neuron_n[NYSCL][NXSCL][Nn]/*,
                                size_t pitch*/) {
  const int t = blockIdx.z * blockDim.z + threadIdx.z;
  VTYPE sum = 0;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // sliding window;
  for (int ky = 0; ky < Ky; ky++) {
    for (int kx = 0; kx < Kx; kx++) {
      for (int i = 0; i < Ni; i++) {
        VTYPE sv = synapse[ky][kx][t][i];
        VTYPE nv = neuron_i[ky+y][kx+x][i];
        sum += sv * nv;
      }
    }
  }
  neuron_n[y][x][t] = GPU_transfer(sum);
}

int main(void) {
  std::cout << "------ Running CPU version ------" << std::endl;

  synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])   aligned_alloc(64, SYNAPSE_SIZE * sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni]) aligned_alloc(64, NYPAD * NXPAD * Ni * sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_alloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_alloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));
  neuron_n3 = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_alloc(64, NYSCL * NXSCL * Nn * sizeof(VTYPE));

  fill_random((VTYPE *) synapse, SYNAPSE_SIZE);
  fill_random((VTYPE *) neuron_i, NXPAD * NYPAD * Ni);

  std::cout << "Simple version: \t";
  auto f1 = std::bind(convolution, *synapse, *neuron_i, *neuron_n);
  timeit(f1);

  std::cout << "Tiled version:\t";  
  auto f2 = std::bind(convolution_tiled, *synapse, *neuron_i, *neuron_n2);
  timeit(f2);

  compare((VTYPE *)neuron_n, (VTYPE *)neuron_n2, NYSCL * NXSCL * Nn);

  std::cout << "------ Running GPU version ------" << std::endl;


/*
  // flatten synapse from 4D into 3D array
  cudaExtent extent_synapse = make_cudaExtent(Kx * sizeof(VTYPE), Ky, Nn * Ni);
  cudaExtent extent_neuron_i = make_cudaExtent(NXPAD * sizeof(VTYPE), NYPAD, Ni);
  cudaExtent extent_neuron_n = make_cudaExtent(NXSCL * sizeof(VTYPE), NYSCL, Ni);

  size_t pitch_synapse, pitch_neuron_i, pitch_neuron_n;
  cudaPitchedPtr d_synapse, d_neuron_i, d_neuron_n;
  cudaMalloc3D(&d_synapse, extent_synapse);
  cudaMalloc3D(&d_neuron_i, extent_neuron_i);
  cudaMalloc3D(&d_neuron_n, extent_neuron_n);

  cudaMemcpy3DParms p_synapse = {}, p_neuron_i = {}, p_neruron_n = {};
  p_synapse.srcPtr = synapse;
  p_synapse.dstPtr = d_synapse;
  p_synapse.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(p_synapse);
*/
  /* use oridnary memory layout for the naive version now */
  VTYPE (*d_synapse) [Kx][Nn][Ni];
  VTYPE (*d_neuron_i) [NXPAD][Ni];
  VTYPE (*d_neuron_n) [NXSCL][Nn];
  cudaMalloc(&d_synapse, Kx * Ky * Nn * Ni * sizeof(VTYPE));
  cudaMalloc(&d_neuron_i, NXPAD * NYPAD * Ni * sizeof(VTYPE));
  cudaMalloc(&d_neuron_n, NXSCL * NYSCL * Ni * sizeof(VTYPE));
  cudaMemcpy(d_synapse, synapse, Kx * Ky * Nn * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_neuron_i, neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice); 

  memset(neuron_n3, 0, NYSCL * NXSCL * Nn * sizeof(VTYPE));

  std::cout << "Simple version: \t";

//  for (int num_threads = 2; num_threads <= 1024; num_threads *= 2) {
//    int num_blocks = Nn / num_threads;
  
    cudaMemset(d_neuron_n, 0, NYSCL * NXSCL * Nn * sizeof(VTYPE));
    CUDA_timeit([&]() {
      dim3 grid_size(NXSCL/Tx, NYSCL/Ty, Nn/Tn);
      dim3 block_size(Tx, Ty, Tn);
//      printf("%d %d %d %d %d %d\n", NXSCL/Tx, NYSCL/Ty, Nn/Tn, Tx, Ty, Tn);
      GPU_convolution<<<grid_size, block_size>>>(d_synapse, d_neuron_i, d_neuron_n/*, pitch*/);
    });
    cudaMemcpy(neuron_n3, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
  auto err = cudaGetLastError();
  if(err) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    return 0;
  } 
//    std::cout << "#threads = " << num_threads << ", #blocks = " << num_blocks << std::endl;
    compare((VTYPE *) neuron_n, (VTYPE *) neuron_n3, NYSCL * NXSCL * Nn);
//  }
  std::cout << "Tiled version:\t";  
}
