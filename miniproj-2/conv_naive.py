import math
import csv

#Maximum of TFLOPs for Titan V (13.87 TFlops)
MAXTOPS = 14.0 * 10e12
# Memory Bandwidth for Titan V (653 GB/s)
MEM_BW = 650 * (2 ** 30)
# Memory Cloak for Titan V (1.7Gbps)
MEM_CLOCK = 1.7
# Cache maxsize for CUDA  (256 MB for Integer/ 4GB for max)
CACHE_MAX = 4608 * (2**10)

Element_size = 2

FILENAME = 'clean_conv.csv'

W_i, H_i, C_i, B = -1, -1, -1, -1
W_f, H_f, C_o, PAD, STRI = -1, -1, -1, -1, -1
W_o, H_o, M, N, K = -1, -1, -1, -1, -1

def conv_comp_time ():
    total_ops = W_f * W_o * H_f * H_o * C_i * C_o * B
    return total_ops / MAXTOPS

def conv_mem_time ():
    total_mem_read  = (W_i * H_i * C_i * B + W_f * H_f * C_o)
    total_mem_store = (W_o * H_o * C_o * B)
    total_mem_rw = (total_mem_read + total_mem_store) * Element_size
    return total_mem_rw / MEM_BW

def calculate_time ():
    compute_time = conv_comp_time() * 10e6
    memory_time = conv_mem_time() * 10e6
    max_value = max(compute_time, memory_time)
    # print ('\t%f' % compute_time, '\t%f' % memory_time, '\t%f' % max(compute_time, memory_time))
    return max_value

def read_parameters(file_name):
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        line_count = 0
        parameters = list (reader)
    return parameters

def set_parameters (w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o, m, n, k):
    global W_i, H_i, C_i, B, W_f, H_f, C_o, PAD, STRI, W_o, H_o, M, N, K
    W_i, H_i, C_i, B = w_i, h_i, c_i, b
    W_f, H_f, C_o, PAD, STRI = w_f, h_f, c_o, padding, stride
    W_o, H_o, M, N, K = w_o, h_o, m, n, k

if __name__ == '__main__':
    parameters = read_parameters(FILENAME)

    for parameter in parameters:
        w_i, h_i, c_i, b = int(parameter[0]), int(parameter[1]), int(parameter[2]), int(parameter[3])
        w_f, h_f, c_o, padding, stride = int(parameter[5]), int(parameter[6]), int(parameter[4]), int(parameter[7]), int(parameter[9])
        w_o = math.floor(w_i + 2 * padding - w_f) / stride
        h_o = math.floor(h_i + 2 * padding - h_f) / stride
        m = b * h_o * w_o
        n = c_o
        k = c_i * h_f * w_f
        block_m, block_n, block_k = 128, 128, 64
        set_parameters (w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o, m, n, k)
        # print (w_i, h_i, c_i, b, w_f, h_f, c_o, padding, stride, w_o, h_o, m, n, k)
        bound_time = calculate_time()



    '''w_i, h_i, c_i, b = 700, 161, 1, 32
    w_f, h_f, c_o, padding, stride = 5, 20, 32, 0, 2
    w_o = math.floor(w_i + 2 * padding - w_f) / stride
    h_o = math.floor(h_i + 2 * padding - h_f) / stride
    m = b * h_o * w_o
    n = c_o
    k = c_i * h_f * w_f
    block_m, block_n, block_k = 128, 128, 64 '''

    # bound_time = calculate_time()
    # print('Bounded at %f' % bound_time)
