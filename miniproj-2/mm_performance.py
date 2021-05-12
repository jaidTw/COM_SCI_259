import numpy as np

#Maximum of TFLOPs ()
MAXTOPS = 14.0
# Memory Bandwidth for Titan V (653 GB/s)
MEM_BW = 650
# Memory Cloak for Titan V (1.7Gbps)
MEM_CLOCK = 1.7
# Cache maxsize for CUDA  (256 MB for Integer/ 4GB for max)
CACHE_MAX = 256
# L2 Cache

def mm_comp_time (m, n, k):
    total_ops = m * n * k
    return total_ops/(MAXTOPS * 10e6)

def mm_mem_time (m, n, k):
    total_mem_read = ( m * n + n * k + m * k) / (CACHE_MAX * 1024 * 1024) 
    return (total_mem_read/MEM_BW) * 10e6

def calculate_time (m, n, k): 
    compute_time = mm_comp_time(m, n, k)
    memory_time = mm_mem_time(m, n, k)
    #max_value = max(compute_time, mm_time)
    print ('%d\t%d\t%d\t' % (m, n, k), '\t%f' % compute_time, '\t%f' % memory_time, '\t%f' % max(compute_time, memory_time))

if __name__ == '__main__':
    print ('m\tn\tk\tcomputation bound\tmemory bound\tpredicted time')
    calculate_time(5124,  700, 2048)
    calculate_time(  35,  700, 2048)
    calculate_time(5124,  700, 2560)
    calculate_time(  35,  700, 2560)
    calculate_time(5124, 1500, 2048)
    calculate_time(  35, 1500, 2048)
    calculate_time(5124, 1500, 2560)
    calculate_time(  35, 1500, 2560)
    # mm_mem_time(1760, 16, 1760)
