#include "dnn.hpp"

int main(void) {
    int arr[2][5][3] = {
        {
            {0, 1, 2},
            {3, 4, 5},
            {6, 7, 8},
            {9, 10, 11},
            {12, 13 ,14}
        },
        {
            {15, 16, 17},
            {18, 19, 20},
            {21, 22, 23},
            {24, 25, 26},
            {27, 28, 29}
        }
    };

    cudaExtent ext = make_cudaExtent(5 * sizeof();

}
