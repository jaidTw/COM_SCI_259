import numpy as np
import csv

#Maximum of TFLOPs for Titan V (13.87 TFlops)
MAXTOPS = 14.0
# Memory Bandwidth for Titan V (653 GB/s)
MEM_BW = 650
# Memory Cloak for Titan V (1.7Gbps)
MEM_CLOCK = 1.7
# Cache maxsize for CUDA  (256 MB for Integer/ 4GB for max)
CACHE_MAX = 256
# L2 Cache

FILENAME = 'gemm.csv'

def mm_comp_time (m, n, k):
    total_ops = m * n * k
    return total_ops/(MAXTOPS * 10e6)

def mm_mem_time (m, n, k):
    total_mem_read = ( m * n + n * k + m * k) / (CACHE_MAX * 1024 * 1024) 
    return (total_mem_read/MEM_BW) * 10e6

def calculate_time (m, n, k): 
    compute_time = mm_comp_time(m, n, k)
    memory_time = mm_mem_time(m, n, k)
    max_value = max(compute_time, memory_time)
    #print ('%d\t%d\t%d\t' % (m, n, k), '\t%f' % compute_time, '\t%f' % memory_time, '\t%f' % max(compute_time, memory_time))
    return max_value

def read_parameters(file_name):
    with open(file_name) as csv_file:
        reader = csv.DictReader(csv_file)
        line_count = 0
        for row in reader:
            if line_count == 0:
                #print(f'{"    ".join(row)}')
                print('m\tn\tk\tDeepBench Time\t Predicted Time')
                line_count += 1
            time = calculate_time(int(row["m"]), int(row["n"]), int(row["k"]))
            print(f'{row["m"]}\t{row["n"]}\t{row["k"]}\t{row["time"]}\t', '\t%f' % time)
            line_count += 1
        print(f'Processed {line_count} lines.')

if __name__ == '__main__':
    read_parameters(FILENAME)
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
