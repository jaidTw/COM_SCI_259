#pragma once

#include <random>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <functional>
#include <iomanip>

#define VTYPE float

inline void cuda_check_error() {
  auto err = cudaGetLastError();
  if(err) {
    std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    exit(0);
  }
}

void fill_random(VTYPE *array, size_t size) {
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_real_distribution<> dis(-0.5, 0.5);
  for(size_t i = 0; i < size; ++i) {
    array[i] = dis(gen);
  }
}

__attribute__ ((noinline)) void timeit(std::function<void ()> f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
  std::cout << std::left << std::setw(10) << diff.count() << " sec(s) elapsed." << std::endl;
}

template <typename F>
__attribute__ ((noinline)) void CUDA_timeit(F f) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  f();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float exec_time;
  cudaEventElapsedTime(&exec_time, start, stop);
  std::cout << std::left << std::setw(10) << exec_time / 1000.0 << " sec(s) elapsed." << std::endl;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// Is this a leaky relu?
VTYPE transfer(VTYPE i) {
  return (i > 0) ? i : i / 4;
}

__device__ VTYPE GPU_transfer(VTYPE i) {
  return (i > 0) ? i : i / 4;
}

void compare(VTYPE* neuron1, VTYPE* neuron2, size_t size) {
  bool error = false;
  for (size_t i = 0; i < size; ++i) {
      if (std::abs(neuron1[i] - neuron2[i]) > 0.001f) {
      error = true; 
      break;
    }
  }
  if (error) {
    for(size_t i = 0; i < size; ++i) {
      std::cout << i << " " << neuron1[i] << ":" << neuron2[i];

      if (std::abs(neuron1[i] - neuron2[i]) > 0.001f)
        std::cout << " \t\tERROR";
      std::cout << "\n";
    }
  }/* else {
    std::cout << "results match" << std::endl;
  }*/
}
