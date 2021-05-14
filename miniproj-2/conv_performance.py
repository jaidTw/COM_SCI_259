import numpy as np
import csv
import math

#Maximum of TFLOPs for Titan V (13.87 TFlops)
MAXTOPS = 14.0
# Memory Bandwidth for Titan V (653 GB/s)
MEM_BW = 650
# Memory Cloak for Titan V (1.7Gbps)
MEM_CLOCK = 1.7
# Cache maxsize for CUDA  (256 MB for Integer/ 4GB for max)
CACHE_MAX = 256 * 1024 * 1024
# L2 Cache Bandwidth (67.5 GB/s/SM)
L1_BW = 67.5 * 1024 * 1024 * 1024
# L1 Cache Bandwidth (1600 GB/s)
L2_BW = 1600 * 1024 * 1024 *1024
# number of SM
SM_NUM = 70
## Titan V: 2048 (threads) / 32 (warp size) = 64 (number of warp)
# Size of a Warp
WARP_SIZE = 32
# Number of warps
WARP_NUM = 64

FILENAME = 'conv.csv'
# block m, block n, block k
CTA_TILE_SELECTION = [(128, 128, 8), (128, 64, 4), (128, 32, 4)]

W_i, H_i, C_i, B = -1, -1, -1, -1
W_f, H_f, C_o, PAD, STRI = -1, -1, -1, -1, -1
W_o, H_o, M, N, K = -1, -1, -1, -1, -1

BLOCK_m, BLOCK_n, BLOCK_k = -1, -1, -1

def l1_cache_traffic ():
    # general memory access inefficiency
    ratio = (W_i + 2 * PAD) * STRI
    ratio = ratio / (W_i + 2 * PAD - W_f + 1)
    MLI_IFmap = ratio
    MLI_Filter = 2.0 # TODO not quite sure here
    T_l1 = M * K * MLI_IFmap + N * K * MLI_Filter
    return T_l1

def l2_cache_traffic ():
    # Intra-CTA spatial Locality
    ratio = (W_i + 2 * PAD) * STRI
    ratio = ratio / (W_i + 2 * PAD - W_f + 1)
    # Distance between the smallest and the largest address
    Dist_v = BLOCK_m * ratio # vertical distance
    Dist_h = ((BLOCK_k - 1) / W_f) * ((W_i - W_f + 1) + STRI * (W_f - BLOCK_k + 1))
    Dist_h = Dist_h + ((W_f - BLOCK_k + 1) / W_f) * (STRI * (BLOCK_k - 1)) # horizontal distance
    A_Dist_v = Dist_v * (BLOCK_k / (H_f * W_f))
    h_mul_w = (H_i + 2 * PAD - H_f + 1) / STRI
    A_Dist_h = Dist_h * (1 + (BLOCK_m / (h_mul_w) ** 2))
    CTA_num = (M / BLOCK_m) * (N / BLOCK_n) * (K / BLOCK_k)
    T_l2 = (A_Dist_v + A_Dist_h) * (K / BLOCK_k) * CTA_num
    # TODO really not sure the value of A_Dist_IFmap & Dist_Filter & cta_num & conv_layer
    return T_l2

def dram_traffic (CTA_col , CTA_tiles_array):
    # T_DRAM_IFmap = B * H_i * W_i * C_i * (C_o / (W_i * H))  #TODO not sure
    T_DRAM_IFmap = (M * BLOCK_k) / (K * BLOCK_m) #TODO not sure
    T_DRAM_Filter = (C_i * H_f * W_f) * C_o
    T_DRAM_total = T_DRAM_IFmap + T_DRAM_Filter
    return T_DRAM_total

def global_load_stream (L1_traffic, L2_traffic, DRAM_traffic):
    #TODO pipeline latency
    latency_l1, latency_l2, latency_dram = 4, 4, 4 
    l1_gls = (latency_l1 + L1_traffic / L1_BW )
    l2_gls = (latency_l2 + L2_traffic / (L2_BW / SM_NUM))
    l3_gls = (latency_dram + DRAM_traffic / (MEM_BW / SM_NUM))
    GLS_traffic = max (l1_gls , l2_gls , l3_gls)
    return GLS_traffic

def shared_memory_access_stream ():
    load_traffic  = (BLOCK_m + BLOCK_n) * BLOCK_k / CACHE_MAX 
    store_traffic = (BLOCK_m + BLOCK_n) * BLOCK_k * WARP_NUM / CACHE_MAX 
    SAS_traffic = load_traffic + store_traffic
    return SAS_traffic

def compute_stream ():
    total_ops = BLOCK_m * BLOCK_n * BLOCK_k
    CS_traffic = total_ops / (MAXTOPS * 10e6)
    return CS_traffic

def calculate_time (m, n, k): 
    compute_time = mm_comp_time(m, n, k)
    memory_time = mm_mem_time(m, n, k)
    max_value = max(compute_time, memory_time)
    #print ('%d\t%d\t%d\t' % (m, n, k), '\t%f' % compute_time, '\t%f' % memory_time, '\t%f' % max(compute_time, memory_time))
    return max_value

def set_parameters (w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o, m, n, k):
    global W_i, H_i, C_i, B, W_f, H_f, C_o, PAD, STRI, W_o, H_o, M, N, K
    W_i, H_i, C_i, B = w_i, h_i, c_i, b
    W_f, H_f, C_o, PAD, STRI = w_f, h_f, c_o, padding, stride
    W_o, H_o, M, N, K = w_o, h_o, m, n, k

def read_parameters(file_name):
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        line_count = 0
        parameters = list (reader)
        '''
        for row in reader:
            if line_count == 0:
                print(f'{"    ".join(row)}')
            #time = calculate_time(int(row["m"]), int(row["n"]), int(row["k"]))
            print(f'{row["w"]}\t{row["h"]}\t{row["c"]}\t{row["n"]}\t')
            line_count += 1
        parameters_dict = {rows[0]:rows[1] for rows in reader}
        '''
        # print(data)
    return parameters

if __name__ == '__main__':
    parameters = read_parameters(FILENAME)
    #print(parameters[1][0])
    w_i, h_i, c_i, b = 700, 161, 1, 32
    w_f, h_f, c_o, padding, stride = 5, 20, 32, 0, 2
    w_o = math.floor(w_i + 2 * padding - w_f) / stride
    h_o = math.floor(h_i + 2 * padding - h_f) / stride
    m = b * h_o * w_o
    n = c_o
    k = c_i * h_f * w_f
    block_m, block_n, block_k = 128, 128, 64
    
    set_parameters(w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o, m, n, k)

    L1_Traffic = l1_cache_traffic()
    L2_Traffic = l2_cache_traffic()
    DRAM_Traffic = dram_traffic(4,2)
    
    GLS = global_load_stream(L1_Traffic, L2_Traffic, DRAM_Traffic)
    SAS = shared_memory_access_stream()
    CS = compute_stream()

    print("%f, %f, %f" % (GLS, SAS, CS))

    '''
    print ('m\tn\tk\tcomputation bound\tmemory bound\tpredicted time')
    calculate_time(5124,  700, 2048)
    calculate_time(  35,  700, 2048)
    calculate_time(5124,  700, 2560)
    calculate_time(  35,  700, 2560)
    calculate_time(5124, 1500, 2048)
    calculate_time(  35, 1500, 2048)
    calculate_time(5124, 1500, 2560)
    calculate_time(  35, 1500, 2560)
    '''# mm_mem_time(1760, 16, 1760)
