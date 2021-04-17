#pragma once

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <functional>

#define VTYPE float

__attribute__ ((noinline)) void timeit(std::function<void ()> f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
  std::cout << "elapsed (sec): " << diff.count() << std::endl;
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
  } else {
    std::cout << "results match" << std::endl;
  }
}
