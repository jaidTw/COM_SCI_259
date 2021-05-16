#!/usr/bin/env python3

import math
import csv
import sys

#Maximum of TFLOPs for Titan V (13.87 TFlops)
MAXTOPS = 13.87 * 10e12
# Memory Bandwidth for Titan V (653 GB/s)
MEM_BW = 652.8 * (2 ** 30)
# L2 Cache Size (4608 KB)
L2_SIZE = 4608 * (2**10)
# L1 Cache Bandwidth (67.5 GB/s/SM)
L1_BW = 67.5 * 2**30
# L2 Cache Bandwidth (1600 GB/s)
L2_BW = 1600 * 2**30
SM_NUM = 70
## Titan V: 2048 (threads) / 32 (warp size) = 64 (number of warp)
WARP_SIZE = 32
WARP_NUM = 64

DATA_SIZE = 4

W_i, H_i, C_i, B = -1, -1, -1, -1
W_f, H_f, C_o, PAD, STRI = -1, -1, -1, -1, -1
W_o, H_o, M, N, K = -1, -1, -1, -1, -1

BLOCK_m, BLOCK_n, BLOCK_k = -1, -1, -1

def conv_comp_time_naive():
    total_ops = W_f * W_o * H_f * H_o * C_i * C_o * B
    return total_ops / MAXTOPS

def conv_mem_time_naive():
    total_mem_read  = (W_i * H_i * C_i * B + W_f * H_f * C_o)
    total_mem_write = (W_o * H_o * C_o * B)
    total_mem_rw = (total_mem_read + total_mem_write) * DATA_SIZE
    return total_mem_rw / MEM_BW

def calculate_time_naive ():
    return max(conv_comp_time_naive(), conv_mem_time_naive()) * 10e6

def get_cta_tile_size(c_o):
    if c_o >= 310:
        return 128, 128, 8
    elif c_o >= 240:
        return 128, 64, 4
    elif c_o >= 180:
        return 128, 128, 8
    elif c_o >= 110:
        return 128, 64, 4
    elif c_o >= 80:
        return 128, 128, 8
    elif c_o >= 50:
        return 128, 32, 4
    elif c_o >= 15:
        return 128, 64, 4
    else:
        return 128, 32, 4

def get_l1_cache_traffic ():
    # general memory access inefficiency
    MLI_IFmap = (W_i + 2 * PAD) * STRI / (W_i + 2 * PAD - W_f + 1)
    col_elems = WARP_SIZE / BLOCK_k
    MLI_Filter = ((col_elems - 1) * 2 + (8 - (col_elems - 1))) / col_elems
    return M * K * MLI_IFmap + N * K * MLI_Filter

def get_l2_cache_traffic ():
    # Intra-CTA spatial Locality
    ratio = (W_i + 2 * PAD) * STRI
    ratio = ratio / (W_i + 2 * PAD - W_f + 1)
    # Distance between the smallest and the largest address
    Dist_v = BLOCK_m * ratio # vertical distance
    Dist_h = ((BLOCK_k - 1) / W_f) * ((W_i - W_f + 1) + STRI * (W_f - BLOCK_k + 1))
    Dist_h += ((W_f - BLOCK_k + 1) / W_f) * (STRI * (BLOCK_k - 1)) # horizontal distance
    A_Dist_v = Dist_v * (BLOCK_k / (H_f * W_f))
    h_mul_w = (H_i + 2 * PAD - H_f + 1) / STRI
    A_Dist_h = Dist_h * (1 + (BLOCK_m / (h_mul_w) ** 2))
    CTA_num = (M / BLOCK_m) * (N / BLOCK_n) # ceiling?
    conv_layer = 1 # TODO: not sure
    return (abs(A_Dist_v) + abs(A_Dist_h) + BLOCK_k) * (K / BLOCK_k) * (CTA_num / conv_layer)

def get_dram_traffic ():
    T_DRAM_IFmap = B * H_i * W_i * C_i * ((C_o / BLOCK_n)/((C_o / BLOCK_n) * (H_o * W_o * B / BLOCK_m)))
    T_DRAM_Filter = C_i * H_f * W_f * C_o
    return (T_DRAM_IFmap + T_DRAM_Filter) * DATA_SIZE

def get_global_load_stream (L1_traffic, L2_traffic, DRAM_traffic):
    # TODO: pipeline latency
    latency_l1, latency_l2, latency_dram = 4, 4, 4 
    l1_gls = (latency_l1 + L1_traffic / L1_BW )
    l2_gls = (latency_l2 + L2_traffic / (L2_BW / SM_NUM))
    mem_gls = (latency_dram + DRAM_traffic / (MEM_BW / SM_NUM))
    return max (l1_gls , l2_gls , mem_gls)

def get_shared_memory_access_stream ():
    # TODO: CACHE_MAX -> BW LD/ST
    load_traffic  = (BLOCK_m + BLOCK_n) * BLOCK_k / 1
    # TODO: BLKwM * blkWN (wrap tiles)
    store_traffic = (BLOCK_m + BLOCK_n) * BLOCK_k * WARP_NUM / 1
    return load_traffic + store_traffic

def get_compute_stream ():
    total_ops = BLOCK_m * BLOCK_n * BLOCK_k
    return total_ops / MAXTOPS

def get_SM_execution_time():
    tPrologue = 1
    tEpilogue = 1
    tMAC = 1
    tDRAM_LAT = 1
    tMEM_BW = 1

def calculate_time ():
    return max_value

def set_parameters (w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o, m, n, k):
    global W_i, H_i, C_i, B, W_f, H_f, C_o, PAD, STRI, W_o, H_o, M, N, K, BLOCK_m, BLOCK_n, BLOCK_k
    W_i, H_i, C_i, B = w_i, h_i, c_i, b
    W_f, H_f, C_o, PAD, STRI = w_f, h_f, c_o, padding, stride
    W_o, H_o, M, N, K = w_o, h_o, m, n, k
    BLOCK_m, BLOCK_n, BLOCK_k = get_cta_tile_size(c_o)

def load_parameters(file_name):
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        return list(reader)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input>", file=sys.stderr)
        exit()

    parameters = load_parameters(sys.argv[1])

    RMSE_error_naive = 0
    RMSE_error_l1 = 0
    print("%5s %5s %4s %3s  %3s %3s %5s %4s %7s %5s %5s  %10s %5s %10s %5s" %
            ("W_i", "H_i", "C_i", "B", "W_f", "H_f", "C_o", "Pad", "Stride", "W_o", "C_o", "Naive (us)", "err", "L1 (us)", "err"))
    for row in parameters:
        w_i, h_i, c_i, b, c_o, w_f, h_f, padding, _, stride = list(map(int, row[:10]))
        real_time = float(row[15])
        w_o = math.floor(w_i + 2 * padding - w_f) / stride
        h_o = math.floor(h_i + 2 * padding - h_f) / stride
        m = b * h_o * w_o
        n = c_o
        k = c_i * h_f * w_f
        
        set_parameters(w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o, m, n, k)
        exec_time = calculate_time_naive()

        L1_Traffic = get_l1_cache_traffic()
        L2_Traffic = get_l2_cache_traffic()
        DRAM_Traffic = get_dram_traffic()
#        print("%15.4f, %15.4f, %15.4f, %15.4f"% (L1_Traffic/L1_BW*10e6, L2_Traffic/L2_BW*10e6, DRAM_Traffic/MEM_BW*10e6, conv_mem_time_naive()*10e6))
        normal_l1 = L1_Traffic / (L1_BW * SM_NUM)
        normal_l2 = L2_Traffic / L2_BW
        normal_dram = DRAM_Traffic / MEM_BW
        l1_time = L1_Traffic / MEM_BW * 10e6

        error_naive = abs((real_time - exec_time) / real_time)
        error_l1 = abs((real_time - l1_time) / real_time)
        RMSE_error_naive += (real_time - exec_time) ** 2
        RMSE_error_l1 += (real_time - l1_time) ** 2

        print("%5d %5d %4d %3d  %3d %3d %5d %4d %7d %5d %5d  %10.4f %5.3f %10.4f %5.3f" % (w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o, exec_time, error_naive, l1_time, error_l1))
    RMSE_error_naive = (RMSE_error_naive / len(parameters)) ** 0.5
    RMSE_error_l1 = (RMSE_error_l1 / len(parameters)) ** 0.5
    print(f"RMSE naive: {RMSE_error_naive}")
    print(f"RMSE L1: {RMSE_error_l1}")

