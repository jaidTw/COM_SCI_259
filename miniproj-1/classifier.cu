#include <iostream>
#include <functional>
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
  //#define Tn 5
  //#define Ti 25
  #define Tn 16
  #define Ti 16
#endif

VTYPE synapse[Nn][Ni] __attribute__((aligned(64)));
VTYPE neuron_i[Ni] __attribute__((aligned(64)));
VTYPE neuron_n[Nn] __attribute__((aligned(64)));
VTYPE neuron_n2[Nn] __attribute__((aligned(64)));

void classifier_layer(VTYPE synapse[Nn][Ni],
                      VTYPE neuron_i[Ni],
                      VTYPE neuron_n[Nn]) {
  for (int n = 0; n < Nn; n++) {
    VTYPE temp = 0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = transfer(temp);
  }
}


__global__ void GPU_classifier_layer(VTYPE synapse[Nn][Ni],
                                     VTYPE neuron_i[Ni],
                                     VTYPE neuron_n[Nn]) {
  for (int n = 0; n < Nn; n++) {
    VTYPE temp = 0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = GPU_transfer(temp);
  }
}

void classifier_layer_blocked(VTYPE synapse[Nn][Ni],
                              VTYPE neuron_i[Ni], 
                              VTYPE neuron_n[Nn]) {
  VTYPE sum[Nn] = {};
  for (int outer_n = 0; outer_n < Nn; outer_n += Tnn) { // tiling for output neurons;
    for (int outer_i = 0; outer_i < Ni; outer_i += Tii) { // tiling for input neurons;
      for (int inner_n = outer_n; inner_n < outer_n + Tnn; inner_n += Tn) {
        for (int inner_i = outer_i; inner_i < iii + Tii; inner_i += Ti) {
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

int main(void) {
  std::cout << "------ Running CPU version ------" << std::endl;

  fill_random((VTYPE *) synapse, Nn * Ni);
  fill_random((VTYPE *) neuron_i, Ni);
  memset(neuron_n, 0, Nn * sizeof(VTYPE));
  memset(neuron_n2, 0, Nn * sizeof(VTYPE));

  std::cout << "Simple version: \t";
  auto f1 = std::bind(classifier_layer, synapse, neuron_i, neuron_n);
  timeit(f1);

  std::cout << "Blocked version:\t";  
  auto f2 = std::bind(classifier_layer_blocked, synapse, neuron_i, neuron_n2);
  timeit(f2);

  compare(neuron_n, neuron_n2, Nn);

  std::cout << "------ Running GPU version ------" << std::endl;

  /* TODO: GPU version*/
}

