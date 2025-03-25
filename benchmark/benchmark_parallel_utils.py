"""
Benchmark Utilities

Modified based on the reference from:
https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa

"""

from collections import namedtuple
import time
import jax
import numpy as np
from typing import List

from predtop.parallel import PipeshardParallel
from predtop.stage import PredStageOption
from alpa import (AutoShardingOption, AutoStageOption, AutoLayerOption, ManualStageOption,
                  global_config)
from alpa.timer import timers
from alpa.util import (print_used_time, to_str_round, list_gpu_info)


BenchmarkCase = namedtuple("BenchmarkCase", [
    "batch_size", "model_config", "num_micro_batches", "parallel_mode",
    "parallel_args"
])

LoadSolutionParallelArgs = namedtuple("LoadSolutionParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers",
    "forward_stage_layer_ids", "submesh_physical_shapes",
    "submesh_logical_shapes", "submesh_autosharding_option_dicts"
])

PredictLatParallelArgs = namedtuple("PredictLatParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers",
    "forward_stage_layer_ids", "submesh_physical_shapes",
    "submesh_logical_shapes", "submesh_autosharding_option_dicts", "auto_stage_option"
])

ParallelArgs = namedtuple("SearchParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers", "auto_stage_option"
])


def get_pipeshard_parallel_method(benchmark_case: BenchmarkCase,
                                  allow_mixed_mesh_shape: bool = False,
                                  use_fine_grained_remat: bool = False,
                                  pipeline_schedule: str = "1f1b"):
    """Create the parallel method of a benchmark case.

    Args:
        benchmark_case: The benchmark case.
        num_devices_per_host: The number of devices per host, used in uniform
          parallel mode.
        allow_mixed_mesh_shape: Whether to allow the mixed mesh shape in
          the autosharding pass.
    """

    num_micro_batches = benchmark_case.num_micro_batches
    parallel_mode = benchmark_case.parallel_mode
    parallel_args = benchmark_case.parallel_args
    
    if parallel_mode == "search":
        assert isinstance(parallel_args, ParallelArgs)
        (prefer_reduce_scatter, use_remat, num_auto_layers,
            auto_stage_option) = parallel_args
        add_manual_layer_marker = None
        num_manual_pipeline_stages = None
        add_manual_remat = None
        remat_mode = "coarse_grained_remat" if use_remat else "none"
        auto_stage_option["cached_profile_result"] = None
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            pipeline_schedule=pipeline_schedule,
            layer_option=AutoLayerOption(layer_num=num_auto_layers,
                                            remat_mode=remat_mode),
            stage_option=AutoStageOption(**auto_stage_option))
        
    if parallel_mode == 'predict':
        assert isinstance(parallel_args, PredictLatParallelArgs)
        (prefer_reduce_scatter, use_remat, num_auto_layers,
         forward_stage_layer_ids, submesh_physical_shapes,
         submesh_logical_shapes,
         submesh_autosharding_option_dicts, auto_stage_option) = parallel_args
        add_manual_layer_marker = None
        num_manual_pipeline_stages = None
        add_manual_remat = None
        remat_mode = "coarse_grained_remat" if use_remat else "none"
        auto_stage_option["cached_profile_result"] = None
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            pipeline_schedule=pipeline_schedule,
            layer_option=AutoLayerOption(layer_num=num_auto_layers,
                                            remat_mode=remat_mode),
            stage_option=PredStageOption(
                forward_stage_layer_ids,
                submesh_physical_shapes,
                submesh_logical_shapes,
                submesh_autosharding_option_dicts,
                **auto_stage_option,
            ))
        
        
    elif parallel_mode == "predicts":
        assert isinstance(parallel_args, LoadSolutionParallelArgs)
        (prefer_reduce_scatter, use_remat, num_auto_layers,
         forward_stage_layer_ids, submesh_physical_shapes,
         submesh_logical_shapes,
         submesh_autosharding_option_dicts) = parallel_args
        add_manual_layer_marker = None
        num_manual_pipeline_stages = None
        add_manual_remat = None
        if use_remat:
            remat_mode = ("fine_grained_remat"
                          if use_fine_grained_remat else "coarse_grained_remat")
        else:
            remat_mode = "none"
        model_num_layers = benchmark_case.model_config.num_layers
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            pipeline_schedule=pipeline_schedule,
            layer_option=AutoLayerOption(
                layer_num=num_auto_layers,
                remat_mode=remat_mode,
                fine_grained_remat_layer_num=model_num_layers),
            stage_option=ManualStageOption())
    return (method, add_manual_remat, add_manual_layer_marker,
            num_manual_pipeline_stages)

def benchmark_training_executable(niter,
                                  train_step,
                                  executable,
                                  state,
                                  other_train_step_inputs,
                                  profile_driver_time=False):
    print_used_time(None)

    # Benchmark step time
    warmup = 2 if niter >= 5 else 1

    if profile_driver_time:
        # Benchmark latency with driver overhead
        global_config.use_dummy_value_for_benchmarking = False
        global_config.shard_parallel_sync_for_timer = False
        print("Warmup")
        for i in range(warmup):
            state = train_step(state, *other_train_step_inputs)
        executable.sync()
        niter -= warmup
        print("Benchmark")
        tic = time.time()
        for i in range(niter):
            state = train_step(state, *other_train_step_inputs)
        executable.sync()
        e2e_latency = (time.time() - tic) / niter
        latencies = [e2e_latency]
        print(f"latency with driver overhead: {e2e_latency:.3f}")
    else:
        # Benchmark latency without driver overhead
        for i in range(niter):
            print(f"Iteration {i} ...")
            state = train_step(state, *other_train_step_inputs)
            if isinstance(state, tuple):
                # In case the train_step returns extra info (e.g. loss),
                # Get the actual state out.
                state = state[0]
            executable.sync()

        latencies = executable.get_execution_time_costs()[warmup:]

    print_used_time("Benchmark")

    return latencies


def compile_pipeshard_executable(train_step, state,
                                 other_train_step_inputs, predict=False):
    print_used_time(None)

    compilation_times = {
        k: timers(k).elapsed(mode="sum") for k in [
            "stage-construction", "stage-construction-dp",
            "stage-construction-compilation", "stage-construction-profiling"
        ]
    }
    res = train_step.get_executable(state, *other_train_step_inputs)
    print_used_time("Compile (driver)")

    if predict:
        return res, compilation_times
    
    executable = res
    print(
        f"compilation time breakdown: {to_str_round(compilation_times, 2)}")

    executable.dump_debug_info("tmp")
    executable.sync()
    print_used_time("Compile (worker)")
    return executable, compilation_times

def compile_and_benchmark_pipeshard_training_executable(
        niter,
        train_step,
        state,
        other_train_step_inputs,
        profile_driver_time=False, predict=False):
    res, compilation_times = compile_pipeshard_executable(
        train_step, state, other_train_step_inputs, predict)
    
    if predict:
        return res, None, None, None
    
    latencies = benchmark_training_executable(
        niter,
        train_step,
        res,
        state,
        other_train_step_inputs,
        profile_driver_time=profile_driver_time)
    max_mem_allocated = res.mesh_group.get_max_memory_allocated()

    return latencies, max_mem_allocated, compilation_times, res


def compute_avg_stage_latencies(timelines: List[tuple]):
    stage_latencies = []
    for request_timeline in timelines:
        sorted_timeline = sorted(request_timeline, key=lambda x: x[0])
        stage_borders = [sorted_timeline[0][0]]
        for _, e, _, _ in sorted_timeline:
            stage_borders.append(e)
        stage_latency = [
            stage_borders[i + 1] - stage_borders[i]
            for i in range(len(stage_borders) - 1)
        ]
        stage_latencies.append(stage_latency)
    return np.mean(stage_latencies, axis=0)

def get_num_hosts_and_num_devices():
    """Get the number of hosts and the number of devices per host for benchmark
    scripts."""

    num_hosts = 1
    if global_config.backend == "gpu":
        num_devices_per_host = list_gpu_info().count("UUID")
    elif global_config.backend == "tpu":
        num_devices_per_host = len(jax.devices("tpu"))
    else:
        raise ValueError(
            f"Unsupported backend: {global_config.backend}")
        
    return num_hosts, num_devices_per_host
