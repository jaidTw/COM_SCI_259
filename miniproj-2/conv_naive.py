#!/usr/bin/env python3

import math
import csv
import sys

#Maximum of TFLOPs for Titan V (13.87 TFlops)
MAXTOPS = 13.87 * 10e12
# Memory Bandwidth for Titan V (653 GB/s)
MEM_BW = 652.8 * (2 ** 30)
# L2 Cache Size (4608 KB)
CACHE_MAX = 4608 * (2**10)

DATA_SIZE = 2

FILENAME = 'clean_conv.csv'

W_i, H_i, C_i, B = -1, -1, -1, -1
W_f, H_f, C_o, PAD, STRI = -1, -1, -1, -1, -1
W_o, H_o = -1, -1

def conv_comp_time ():
    total_ops = W_f * W_o * H_f * H_o * C_i * C_o * B
    return total_ops / MAXTOPS

def conv_mem_time ():
    total_mem_read  = (W_i * H_i * C_i * B + W_f * H_f * C_o)
    total_mem_write = (W_o * H_o * C_o * B)
    total_mem_rw = (total_mem_read + total_mem_write) * DATA_SIZE
    return total_mem_rw / MEM_BW

def calculate_time ():
    return max(conv_comp_time(), conv_mem_time()) * 10e6

def load_parameters(file_name):
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        return list(reader)

def set_parameters (w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o):
    global W_i, H_i, C_i, B, W_f, H_f, C_o, PAD, STRI, W_o, H_o
    W_i, H_i, C_i, B = w_i, h_i, c_i, b
    W_f, H_f, C_o, PAD, STRI = w_f, h_f, c_o, padding, stride
    W_o, H_o = w_o, h_o

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input>", file=sys.stderr)
        exit()

    parameters = load_parameters(sys.argv[1])

    print("%5s %5s %4s %3s  %3s %3s %5s %4s %7s %5s %5s  %9s" %
            ("W_i", "H_i", "C_i", "B", "W_f", "H_f", "C_o", "Pad", "Stride", "W_o", "C_o", "Time (us)"))
    for row in parameters:
        w_i, h_i, c_i, b, c_o, w_f, h_f, padding, _, stride = list(map(int, row[:10]))
        w_o = math.floor(w_i + 2 * padding - w_f) / stride
        h_o = math.floor(h_i + 2 * padding - h_f) / stride
        set_parameters (w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o)
        exec_time = calculate_time()
        print("%5d %5d %4d %3d  %3d %3d %5d %4d %7d %5d %5d  %9f" % (w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o, exec_time))
