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

__global__ void GPU_convolution_pitch(cudaPitchedPtr p_synapse, 
                                      cudaPitchedPtr p_neuron_i, 
                                      cudaPitchedPtr p_neuron_n) {
  char *synapse = (char *)p_synapse.ptr;
  char *neuron_i = (char *)p_neuron_i.ptr;
  char *neuron_n = (char *)p_neuron_n.ptr;

  const int t = blockIdx.z * blockDim.z + threadIdx.z;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  VTYPE sum = 0;

  // sliding window;
  for (int ky = 0; ky < Ky; ky++) {
    for (int kx = 0; kx < Kx; kx++) {
      for (int i = 0; i < Ni; i++) {
        VTYPE sv = *(VTYPE *)(synapse + p_synapse.pitch * (p_synapse.ysize * ky + kx) + (t * Nn + i) * sizeof(VTYPE));
        VTYPE nv = *(VTYPE *)(neuron_i + p_neuron_i.pitch * (p_neuron_i.ysize * (ky+y) + (kx+x)) + i * sizeof(VTYPE));
        sum += sv * nv;
      }
    }
  }
//  neuron_n[y][x][t] = GPU_transfer(sum);
  *(VTYPE *)(neuron_n + p_neuron_n.pitch * (p_neuron_n.ysize * y + x) + t * sizeof(VTYPE)) = GPU_transfer(sum);
}


void MallocAndCpy3D(cudaPitchedPtr &devPtr, void *src, cudaExtent &extent) {
  cudaMalloc3D(&devPtr, extent);

  cudaMemcpy3DParms params;
  memset(&params, 0, sizeof(params));
  params.srcPtr.pitch = extent.width;
  params.srcPtr.ptr = src;
  params.srcPtr.xsize = extent.width / sizeof(VTYPE);
  params.srcPtr.ysize = extent.height;
  params.dstPtr = devPtr;
  params.kind = cudaMemcpyHostToDevice;
  params.extent = extent;
  cudaMemcpy3D(&params);
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

  std::cout << "Simple version:\t";
  auto f1 = std::bind(convolution, *synapse, *neuron_i, *neuron_n);
  timeit(f1);

  std::cout << "Tiled version: \t";  
  auto f2 = std::bind(convolution_tiled, *synapse, *neuron_i, *neuron_n2);
  timeit(f2);

  compare((VTYPE *)neuron_n, (VTYPE *)neuron_n2, NYSCL * NXSCL * Nn);

  std::cout << "------ Running GPU version ------" << std::endl;

  // Initialization for naive version
  VTYPE (*d_synapse) [Kx][Nn][Ni];
  VTYPE (*d_neuron_i) [NXPAD][Ni];
  VTYPE (*d_neuron_n) [NXSCL][Nn];
  cudaMalloc(&d_synapse, Kx * Ky * Nn * Ni * sizeof(VTYPE));
  cudaMalloc(&d_neuron_i, NXPAD * NYPAD * Ni * sizeof(VTYPE));
  cudaMalloc(&d_neuron_n, NXSCL * NYSCL * Ni * sizeof(VTYPE));
  cudaMemcpy(d_synapse, synapse, Kx * Ky * Nn * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_neuron_i, neuron_i, NYPAD * NXPAD * Ni * sizeof(VTYPE), cudaMemcpyHostToDevice);


  // Initialization for pitch version, flatten synapse from 4D into 3D array
  cudaExtent extent_synapse = make_cudaExtent(Nn * Ni * sizeof(VTYPE), Kx, Ky);
  cudaExtent extent_neuron_i = make_cudaExtent(Ni * sizeof(VTYPE), NXPAD, NYPAD);
  cudaExtent extent_neuron_n = make_cudaExtent(Ni * sizeof(VTYPE), NXSCL, NYSCL);

  cudaPitchedPtr d_p_synapse, d_p_neuron_i, d_p_neuron_n;
  MallocAndCpy3D(d_p_synapse, synapse, extent_synapse);
  MallocAndCpy3D(d_p_neuron_i, neuron_i, extent_neuron_i);
  MallocAndCpy3D(d_p_neuron_n, neuron_n, extent_neuron_n);

  for (int tn = 1; tn <= Nn; tn *= 2) {
  dim3 grid_size(NXSCL/Tx, NYSCL/Ty, Nn/tn);
  dim3 block_size(Tx, Ty, tn);
  std::cout << "Simple version:\t";
  memset(neuron_n3, 0, NYSCL * NXSCL * Nn * sizeof(VTYPE));
  CUDA_timeit([&]() {
    GPU_convolution<<<grid_size, block_size>>>(d_synapse, d_neuron_i, d_neuron_n);
  });
  cuda_check_error();
  cudaMemcpy(neuron_n3, d_neuron_n, NYSCL * NXSCL * Nn * sizeof(VTYPE), cudaMemcpyDeviceToHost);
  compare((VTYPE *) neuron_n, (VTYPE *) neuron_n3, NYSCL * NXSCL * Nn);

  std::cout << "Pitch version: \t";  
  memset(neuron_n2, 0, NYSCL * NXSCL * Nn * sizeof(VTYPE));
  
  cudaMemset3D(d_p_neuron_n, 0, extent_neuron_n);
  CUDA_timeit([&]() {
    GPU_convolution_pitch<<<grid_size, block_size>>>(d_p_synapse, d_p_neuron_i, d_p_neuron_n);
  });
  cuda_check_error();
  cudaMemcpy3DParms copyback;
  memset(&copyback, 0, sizeof(copyback));
  copyback.dstPtr.pitch = Ni * sizeof(VTYPE);
  copyback.dstPtr.ptr = neuron_n2;
  copyback.dstPtr.xsize = Ni;
  copyback.dstPtr.ysize = NXSCL;
  copyback.srcPtr = d_p_neuron_n;
  copyback.kind = cudaMemcpyDeviceToHost;
  copyback.extent = extent_neuron_n;
  cudaMemcpy3D(&copyback);
  compare((VTYPE *) neuron_n, (VTYPE *) neuron_n2, NYSCL * NXSCL * Nn);
  std::cout << "#Tn = " << tn << std::endl;
  }

}
