"""
Benchmark Utilities

Modified based on the reference from:
https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa

"""
from benchmark_moe import benchmark_moe_3d_internal
from benchmark_gpt import benchmark_gpt_bert_3d_internal
from alpa import (init, global_config)
from alpa.util import disable_tqdm_globally
import multiprocessing as mp

auto_stage_option = {
    "submesh_physical_shape_space": "small_power_of_two",
    "submesh_logical_shape_space": "all",
    "stage_imbalance_tolerance": 1.0,
    "use_hlo_cost_model": False
}

def benchmark_one_case_internal(model,
                                case,
                                niter,
                                num_hosts,
                                num_devices_per_host,
                                profile_driver_time=False,
                                profile_stage_execution_time=False,
                                local=False,
                                disable_tqdm=False):
    if disable_tqdm:
        disable_tqdm_globally()

    # local mode does not support dummy value
    global_config.use_dummy_value_for_benchmarking = not local

    global_config.pipeline_sync_for_timer = True
    if profile_stage_execution_time:
        global_config.collect_trace = True
    init(cluster="ray")

    # Run benchmark
    if model in ["gpt", "bert"]:
        print('Running gpt')
        result = benchmark_gpt_bert_3d_internal(
            model,
            case,
            niter,
            num_hosts,
            num_devices_per_host,
            profile_driver_time=profile_driver_time)
    elif model == "moe":
        result = benchmark_moe_3d_internal(
            case,
            niter,
            num_hosts,
            num_devices_per_host,
            profile_driver_time=profile_driver_time)
    else:
        raise ValueError(f"Invalid model: {model}")

    return result


def benchmark_and_write_to_namespace(result_namespace, *args, **kwargs):
    result = benchmark_one_case_internal(*args, **kwargs)
    result_namespace.result = result


def benchmark_one_case(*args, use_separate_process=False, **kwargs):
    if not use_separate_process:
        return benchmark_one_case_internal(*args, **kwargs)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    result_namespace = manager.Namespace()
    p = ctx.Process(target=benchmark_and_write_to_namespace,
                    args=(result_namespace, *args),
                    kwargs=kwargs)
    p.start()
    p.join()
    if p.exitcode != 0:
        return -1, -1, [-1], -1, None
    return result_namespace.result