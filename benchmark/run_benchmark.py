import time
import argparse
import numpy as np
from alpa.util import (to_str_round, GB)

from benchmark_parallel_utils import (ParallelArgs, BenchmarkCase)
from benchmark import benchmark_one_case, auto_stage_option
from benchmark_moe import moe_specs
from benchmark_gpt import gpt_specs

moe_model = 'moe'
gpt_model = 'gpt'
benchmark_models = [moe_model, gpt_model]

def benchmark(model_type):
    num_hosts = 2
    num_devices_per_host = 2
    
    num_micro_batches = 64
    num_auto_layers = 32
    prefer_reduce_scatter = True
    use_remat = True
    num_iteration = 3
    
    num_gpus = num_hosts * num_devices_per_host
    max_global_batch_size = 1024
    
    if model_type == moe_model:
        model_config = moe_specs['1.3B']
    elif model_type == gpt_model:
        model_config = gpt_specs['1.3B']
    else:
        raise Exception("invalid model type")
    
    parallel_args = ParallelArgs(prefer_reduce_scatter, use_remat,
                               num_auto_layers, auto_stage_option)

    benchmark_case = BenchmarkCase(
        max_global_batch_size,
        model_config,
        num_micro_batches,
        'search',
        parallel_args
    )

    # Run one case
    print("Working on case: {}".format(str(benchmark_case)))
    result = benchmark_one_case(
        model_type,
        benchmark_case,
        num_iteration,
        num_hosts,
        num_devices_per_host)

    (parameter_count, peak_mem, latencies, tflops, metadata) = result

    values = [
        model_type, model_config, num_micro_batches, num_gpus,
        parallel_args, f"{np.mean(latencies):.3f}",
        f"{np.std(latencies):.3f}", f"{parameter_count/1e9:.3f}B",
        f"{tflops:.2f}", f"{peak_mem/GB:.3f}",
        to_str_round(metadata, 2)
    ]
    
    print(values)

    time.sleep(0.1)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        choices=benchmark_models,
                        type=str,
                        required=True)
    args = parser.parse_args()
    benchmark(args.model)