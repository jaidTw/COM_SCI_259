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

__global__ void GPU_convolution(cudaPitchedPtr p_synapse, 
                                cudaPitchedPtr p_neuron_i, 
                                cudaPitchedPtr p_neuron_n,
                                int batch_begin) {
  char *synapse = (char *)p_synapse.ptr;
  char *neuron_i = (char *)p_neuron_i.ptr;
  char *neuron_n = (char *)p_neuron_n.ptr;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  // t = index of Nn dimensino, b = batch id
  int t = z - (z / Nn) * Nn;
  int b = batch_begin + z / Nn;

  VTYPE sum = 0;

  // sliding window;
  for (int ky = 0; ky < Ky; ky++) {
    for (int kx = 0; kx < Kx; kx++) {
      for (int i = 0; i < Ni; i++) {
        // Access M[x][y][z] in M[X][Y][Z]: T elem = *(T *)((char *)ptr + pitch * (Y * x + y) + z * sizeof(T))
        // sv = synapse[b][ky][kx][t][i]
        VTYPE sv = *(VTYPE *)(synapse + p_synapse.pitch * (p_synapse.ysize * (b * Ky + ky) + kx) + (t * Nn + i) * sizeof(VTYPE));
        // nv = neuron_i[b][ky+y][kx+x][i]
        VTYPE nv = *(VTYPE *)(neuron_i + p_neuron_i.pitch * (p_neuron_i.ysize * (b * NYPAD + (ky+y)) + (kx+x)) + i * sizeof(VTYPE));
        sum += sv * nv;
      }
    }
  }
  // neuron_n[y][x][t] = GPU_transfer(sum)
  *(VTYPE *)(neuron_n + p_neuron_n.pitch * (p_neuron_n.ysize * (b * NYSCL + y) + x) + t * sizeof(VTYPE)) = GPU_transfer(sum);
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

  synapse   = (VTYPE (*)[Ky][Kx][Nn][Ni])   aligned_alloc(64, batch * SYNAPSE_SIZE * sizeof(VTYPE));
  neuron_i  = (VTYPE (*)[NYPAD][NXPAD][Ni]) aligned_alloc(64, batch * NYPAD * NXPAD * Ni * sizeof(VTYPE));
  neuron_n  = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_alloc(64, batch * NYSCL * NXSCL * Nn * sizeof(VTYPE));
  neuron_n2 = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_alloc(64, batch * NYSCL * NXSCL * Nn * sizeof(VTYPE));
  neuron_n3 = (VTYPE (*)[NYSCL][NXSCL][Nn]) aligned_alloc(64, batch * NYSCL * NXSCL * Nn * sizeof(VTYPE));

  fill_random((VTYPE *)synapse, batch * SYNAPSE_SIZE);
  fill_random((VTYPE *)neuron_i, batch * NXPAD * NYPAD * Ni);

  std::cout << "CPU Simple version:\t\t\t\t\t";
  timeit([&]() {
    for(int b = 0; b < batch; ++b) {
      convolution(synapse[b], neuron_i[b], neuron_n[b]);
    }
  });

  std::cout << "CPU Tiled version: \t\t\t\t\t";
  timeit([&]() {
    for(int b = 0; b < batch; ++b) {
      convolution_tiled(synapse[b], neuron_i[b], neuron_n2[b]);
    }
  });

  compare((VTYPE *)neuron_n, (VTYPE *)neuron_n2, batch * NYSCL * NXSCL * Nn);

  // Initialization for pitch version, flatten synapse from 4D into 3D array
  cudaExtent extent_synapse = make_cudaExtent(Nn * Ni * sizeof(VTYPE), Kx, batch * Ky);
  cudaExtent extent_neuron_i = make_cudaExtent(Ni * sizeof(VTYPE), NXPAD, batch * NYPAD);
  cudaExtent extent_neuron_n = make_cudaExtent(Ni * sizeof(VTYPE), NXSCL, batch * NYSCL);

  cudaPitchedPtr d_synapse, d_neuron_i, d_neuron_n;
  MallocAndCpy3D(d_synapse, synapse, extent_synapse);
  MallocAndCpy3D(d_neuron_i, neuron_i, extent_neuron_i);
  MallocAndCpy3D(d_neuron_n, neuron_n, extent_neuron_n);
  // End of Initialization

  std::cout << "GPU version:\n";
  for(int tn = 1; tn < 128; tn *= 2) {
    dim3 grid_size(NXSCL/Tx, NYSCL/Ty, (Nn / tn) * batch_in_parallel);
    dim3 block_size(Tx, Ty, tn);
    if(Tx * Ty * tn > 1024 || tn > 64) continue;
    printf("Grid: (%2d, %2d, %4d), Block: (%2d, %2d, %2d), Tn=%2d\t", grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, tn);
    memset(neuron_n2, 0, batch * NYSCL * NXSCL * Nn * sizeof(VTYPE));
    cudaMemset3D(d_neuron_n, 0, extent_neuron_n);
    CUDA_timeit([&]() {
      for(int b = 0; b < batch; b += batch_in_parallel) {
        GPU_convolution<<<grid_size, block_size>>>(d_synapse, d_neuron_i, d_neuron_n, b);
      }
    });
    cuda_check_error();
    cudaMemcpy3DParms copyback;
    memset(&copyback, 0, sizeof(copyback));
    copyback.dstPtr.pitch = Ni * sizeof(VTYPE);
    copyback.dstPtr.ptr = neuron_n2;
    copyback.dstPtr.xsize = Ni;
    copyback.dstPtr.ysize = NXSCL;
    copyback.srcPtr = d_neuron_n;
    copyback.kind = cudaMemcpyDeviceToHost;
    copyback.extent = extent_neuron_n;
    cudaMemcpy3D(&copyback);
    compare((VTYPE *)neuron_n, (VTYPE *)neuron_n2, batch * NYSCL * NXSCL * Nn);
    }
}
