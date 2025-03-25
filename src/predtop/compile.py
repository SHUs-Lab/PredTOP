"""
Compile executables for pipeshard parallelism.

Modified based on the reference from:
https://github.com/alpa-projects/alpa/blob/main/alpa/pipeline_parallel/compile_executable.py

"""

import logging
from typing import Callable, Sequence, Optional

from jax import linear_util as lu
from jax._src.lib import xla_client as xc
from jax.core import gensym, AbstractValue, ClosedJaxpr
from jax.interpreters import pxla
from jax.tree_util import PyTreeDef

from alpa.device_mesh import VirtualPhysicalMesh
from alpa.global_env import global_config
from alpa.pipeline_parallel.pipeshard_executable import PipeshardDriverExecutable
from alpa.pipeline_parallel.runtime_emitter import (
    OverlapFriendlyPipelineInstEmitter, PipelineInstEmitter)
from alpa.pipeline_parallel.schedules import create_pipeline_schedule
from alpa.pipeline_parallel.computation import (create_donation_mapping, split_donate_invars)
from alpa.pipeline_parallel.apply_grad import (process_apply_gradient)
from alpa.pipeline_parallel.layer_construction import LayerOption
from alpa.pipeline_parallel.schedules import gen_dependency_with_stages

from predtop.stage import (
    cluster_layers_and_slice_mesh_pred, get_mesh_models, lat_pred,
    PredStageOption
)

from alpa.pipeline_parallel.stage_construction import (StageOption, ManualStageOption)

from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.shard_parallel.manual_sharding import (ManualShardingOption,
                                                 ParsedManualShardingOption,
                                                 get_flatten_axis_resources)
from alpa.util import (trace_jaxpr_with_micro_batch, GradFuncTransformContext)

from alpa.pipeline_parallel.compile_executable import (
    debug_compilation_time,
    split_and_process_layers,
    slice_apply_grad_for_stage_construction,
    _rewrite_global_outvars_post_concate,
    get_manual_input_output_sharding_specs,
    shard_each_stage,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def compile_pipeshard_executable(
        fun: lu.WrappedFun, in_tree: PyTreeDef,
        out_tree_thunk: Callable[[], PyTreeDef], static_argnums: Sequence[int],
        donated_invars: Sequence[bool], batch_invars: Sequence[bool],
        virtual_mesh: VirtualPhysicalMesh, num_microbatch: int,
        pipeline_schedule: str, default_as_option: AutoShardingOption,
        layer_option: LayerOption, stage_option: StageOption,
        global_input_shardings: Optional[Sequence[pxla.ShardingSpec]],
        stage_input_shardings: Optional[Sequence[Sequence[pxla.ShardingSpec]]],
        manual_shard_options: Optional[ManualShardingOption],
        *avals: Sequence[AbstractValue]):
    """
    Compile a callable for pipeshard parallel which combines
    pipeline parallelism and 2d shard parallelsim.

    Args:
        fun: The function to be parallelized.
        global_input_shardings: Forcibly set sharding specs of global
          input vars.
        stage_input_shardings: Forcibly set sharding specs of input vars of
          each stage.
        manual_sharding_options: pjit style sharding constraints of global input
          vars.
    """
    if global_config.backend == "tpu":
        raise NotImplementedError("Pipeshard Parallel for tpu is not supported")
    debug_compilation_time(None)
    name_base = f"{fun.__name__}_pipeshard_parallel"

    import copy
    funo = copy.deepcopy(fun)
    
    # Apply layer construction to add pipeline markers.
    with GradFuncTransformContext(layer_option.transform):
        if pipeline_schedule == "inference":
            f_backup = fun.f
            fun.f = layer_option.transform(fun.f)

        # Trace the function with a micro batch to get the jaxpr.
        closed_jaxpr, micro_batch_size = trace_jaxpr_with_micro_batch(
            fun, batch_invars, num_microbatch, avals)

        # Trace again with a full batch.
        # The full batch is used to derive the reduction operator across
        # micro batches (e.g., addition, concatenation).
        if num_microbatch > 1:
            for store in fun.stores:
                if store:
                    store.reset()
            full_batch_closed_jaxpr, _ = trace_jaxpr_with_micro_batch(
                fun, batch_invars, 1, avals)
        else:
            full_batch_closed_jaxpr = None

    debug_compilation_time("trace")

    # flatten manual sharding axis resources
    out_tree = out_tree_thunk()
    if manual_shard_options is not None:
        assert global_input_shardings is None
        parsed_ms_option = get_flatten_axis_resources(manual_shard_options,
                                                      in_tree, out_tree)
    else:
        parsed_ms_option = None

    # FIXME: train the models
    comp_res = compile_pipeshard_executable_internal(
        closed_jaxpr, full_batch_closed_jaxpr, micro_batch_size, donated_invars,
        batch_invars, virtual_mesh, num_microbatch, pipeline_schedule,
        default_as_option, stage_option)

    if isinstance(stage_option, PredStageOption):
        # While predicting single latency, print latency and stop
        print('Iteration latency:', comp_res)
        import sys
        sys.exit(0)
        
    for store in funo.stores:
        if store:
            store.reset()

    mesh_models = comp_res
    with GradFuncTransformContext(layer_option.transform):
        # Trace the function with a micro batch to get the jaxpr.
        closed_jaxpr, micro_batch_size = trace_jaxpr_with_micro_batch(
            funo, batch_invars, num_microbatch, avals)

        # Trace again with a full batch.
        # The full batch is used to derive the reduction operator across
        # micro batches (e.g., addition, concatenation).
        for store in funo.stores:
            if store:
                store.reset()
        full_batch_closed_jaxpr, _ = trace_jaxpr_with_micro_batch(
            funo, batch_invars, 1, avals)

    # flatten manual sharding axis resources
    out_tree = out_tree_thunk()
    if manual_shard_options is not None:
        assert global_input_shardings is None
        parsed_ms_option = get_flatten_axis_resources(manual_shard_options,
                                                      in_tree, out_tree)
    else:
        parsed_ms_option = None
    
    pipeshard_config = compile_pipeshard_executable_internal_pred(
        closed_jaxpr, full_batch_closed_jaxpr, micro_batch_size, donated_invars,
        batch_invars, virtual_mesh, num_microbatch, pipeline_schedule,
        default_as_option, stage_option, name_base, global_input_shardings,
        None, stage_input_shardings, parsed_ms_option, mesh_models)

    executable = PipeshardDriverExecutable(
        mesh_group=virtual_mesh.launched_physical_mesh_group,
        pipeshard_config=pipeshard_config,
        num_batch=num_microbatch,
        layer_option=layer_option,
        in_tree=in_tree,
        out_tree=out_tree,
        static_argnums=static_argnums)
    debug_compilation_time("driver executable")
    return executable

def compile_pipeshard_executable_internal(
        closed_jaxpr: ClosedJaxpr,
        full_batch_closed_jaxpr: Optional[ClosedJaxpr], micro_batch_size: int,
        donated_invars: Sequence[bool], batch_invars: Sequence[bool],
        virtual_mesh: VirtualPhysicalMesh, num_microbatch: int,
        pipeline_schedule: str, default_as_option: AutoShardingOption,
        stage_option: StageOption):
    """
    Args:
        fun: The function to be parallelized.
        global_input_shardings: Forcibly set sharding specs of global
          input vars.
        global_output_shardings: Forcibly set sharding specs of global
          output vars.
        stage_input_shardings: Forcibly set sharding specs of input vars of
          each stage.
    """
    global_invars = closed_jaxpr.jaxpr.invars
    gensym_func = gensym([closed_jaxpr.jaxpr])
    inference_mode = (pipeline_schedule == "inference")

    (closed_jaxpr, global_outvars, jax_pipeline_layers, apply_grad_jaxpr,
     microbatch_bound, reduction_vector, post_microbatch_bound,
     accumulator_mapping, acc_grad_invars,
     acc_grad_outvars) = (split_and_process_layers(closed_jaxpr,
                                                   full_batch_closed_jaxpr,
                                                   num_microbatch,
                                                   inference_mode, gensym_func))

    (jax_apply_layers,
     apply_grad_global_info) = slice_apply_grad_for_stage_construction(
         jax_pipeline_layers, apply_grad_jaxpr, microbatch_bound, global_invars,
         global_outvars, donated_invars, accumulator_mapping, gensym_func,
         inference_mode)
    
    if isinstance(stage_option, PredStageOption):
        return lat_pred(
            jax_pipeline_layers, virtual_mesh, accumulator_mapping,
            acc_grad_invars, acc_grad_outvars, num_microbatch, micro_batch_size,
            jax_apply_layers, apply_grad_global_info, pipeline_schedule,
            default_as_option, stage_option
        )
    else:       
        return get_mesh_models(
            jax_pipeline_layers, virtual_mesh, accumulator_mapping,
            acc_grad_invars, acc_grad_outvars, num_microbatch, micro_batch_size,
            jax_apply_layers, apply_grad_global_info, pipeline_schedule,
            default_as_option, stage_option)

def compile_pipeshard_executable_internal_pred(
        closed_jaxpr: ClosedJaxpr,
        full_batch_closed_jaxpr: Optional[ClosedJaxpr], micro_batch_size: int,
        donated_invars: Sequence[bool], batch_invars: Sequence[bool],
        virtual_mesh: VirtualPhysicalMesh, num_microbatch: int,
        pipeline_schedule: str, default_as_option: AutoShardingOption,
        stage_option: StageOption, name_base: str,
        global_input_shardings: Optional[Sequence[pxla.ShardingSpec]],
        global_output_shardings: Optional[Sequence[pxla.ShardingSpec]],
        stage_input_shardings: Optional[Sequence[Sequence[pxla.ShardingSpec]]],
        parsed_manual_sharding_option: Optional[ParsedManualShardingOption],
        mesh_models: []):
    """
    Args:
        fun: The function to be parallelized.
        global_input_shardings: Forcibly set sharding specs of global
          input vars.
        global_output_shardings: Forcibly set sharding specs of global
          output vars.
        stage_input_shardings: Forcibly set sharding specs of input vars of
          each stage.
    """
    global_invars = closed_jaxpr.jaxpr.invars
    gensym_func = gensym([closed_jaxpr.jaxpr])
    inference_mode = (pipeline_schedule == "inference")

    (closed_jaxpr, global_outvars, jax_pipeline_layers, apply_grad_jaxpr,
     microbatch_bound, reduction_vector, post_microbatch_bound,
     accumulator_mapping, acc_grad_invars,
     acc_grad_outvars) = (split_and_process_layers(closed_jaxpr,
                                                   full_batch_closed_jaxpr,
                                                   num_microbatch,
                                                   inference_mode, gensym_func))

    debug_compilation_time("jaxpr operations")

    (jax_apply_layers,
     apply_grad_global_info) = slice_apply_grad_for_stage_construction(
         jax_pipeline_layers, apply_grad_jaxpr, microbatch_bound, global_invars,
         global_outvars, donated_invars, accumulator_mapping, gensym_func,
         inference_mode)

    # Construct pipeline stages by merging layers
    (jax_pipeline_stages, stage_to_mesh, sliced_virtual_meshes,
     manual_stage_option) = cluster_layers_and_slice_mesh_pred(
         jax_pipeline_layers, virtual_mesh, accumulator_mapping,
         acc_grad_invars, acc_grad_outvars, num_microbatch, micro_batch_size,
         jax_apply_layers, apply_grad_global_info, pipeline_schedule,
         default_as_option, stage_option, mesh_models)    

    num_meshes = len(sliced_virtual_meshes)
    debug_compilation_time("stage construction")

    # Process apply_gradient and donation
    num_devices = [vmesh.num_devices for vmesh in sliced_virtual_meshes]
    (sliced_apply_grad_stages, apply_grad_placement,
     global_outvars, allreduce_groups) = process_apply_gradient(
         apply_grad_jaxpr, microbatch_bound, jax_pipeline_stages, stage_to_mesh,
         gensym_func, num_meshes, global_invars, global_outvars, donated_invars,
         False, num_devices)
    jax_all_stages = jax_pipeline_stages + sliced_apply_grad_stages

    donation_mapping = create_donation_mapping(accumulator_mapping,
                                               donated_invars, global_invars,
                                               global_outvars)
    donate_invars_dict, jax_all_stages = split_donate_invars(
        donation_mapping, jax_all_stages, gensym_func)
    global_outvars, concat_vars_mapping = _rewrite_global_outvars_post_concate(
        global_outvars, reduction_vector, microbatch_bound,
        post_microbatch_bound, gensym_func)
    debug_compilation_time("apply grad")

    # Generate pipeline schedule and placement
    dependency, fwd_intermediates = gen_dependency_with_stages(
        jax_pipeline_stages, num_meshes, sliced_apply_grad_stages)
    schedule = create_pipeline_schedule(
        pipeline_schedule,
        dependency=dependency,
        meshes=sliced_virtual_meshes,
        apply_grad_placement=apply_grad_placement,
        num_batch=num_microbatch)

    # Forcibly set the sharding specs of global invars and outvars.
    # FIXME(yonghao): the invar can appear on multiple meshes and thus different
    # sharding specs
    if global_input_shardings:
        assert len(global_input_shardings) == len(global_invars)
        input_sharding_dict = dict(zip(global_invars, global_input_shardings))
    else:
        input_sharding_dict = {}
    if global_output_shardings:
        assert len(global_output_shardings) == len(global_outvars)
        output_sharding_dict = dict(zip(global_outvars,
                                        global_output_shardings))
    else:
        output_sharding_dict = {}
    if parsed_manual_sharding_option is not None:
        assert (global_input_shardings is None and
                global_output_shardings is None)
        (input_sharding_dicts,
         output_sharding_dicts) = get_manual_input_output_sharding_specs(
             jax_all_stages, manual_stage_option.submesh_logical_shapes,
             parsed_manual_sharding_option, global_invars, global_outvars,
             schedule.stage_mesh_mapping, fwd_intermediates)
    else:
        input_sharding_dicts = [input_sharding_dict] * num_meshes
        output_sharding_dicts = [output_sharding_dict] * num_meshes

    # Call auto-sharding pass to shard each stage
    xla_stages, total_flops = shard_each_stage(
        jax_all_stages, sliced_virtual_meshes, schedule, num_meshes,
        accumulator_mapping, global_invars, acc_grad_outvars,
        donate_invars_dict, num_microbatch,
        manual_stage_option.submesh_logical_shapes,
        manual_stage_option.submesh_autosharding_option_dicts,
        default_as_option, input_sharding_dicts, output_sharding_dicts,
        stage_input_shardings, name_base, gensym_func)
    total_flops *= num_microbatch
    debug_compilation_time("shard stages")

    # Launch the physical mesh group
    if virtual_mesh.launched_physical_mesh_group is None:
        virtual_mesh.get_physical_mesh_group(sliced_virtual_meshes)
    debug_compilation_time("launch meshes")

    # Wrap all things into a distributed runtime
    # TODO(yonghao): use virtual mesh instead of launched physical group
    emitter_kwargs = dict(stages=xla_stages,
                          global_invars=global_invars,
                          grad_dummy_invars=accumulator_mapping,
                          global_outvars=global_outvars,
                          concat_vars_mapping=concat_vars_mapping,
                          mesh_group=virtual_mesh.launched_physical_mesh_group,
                          schedule=schedule,
                          is_batch=batch_invars,
                          num_batch=num_microbatch,
                          default_auto_sharding_option=default_as_option,
                          manual_stage_option=manual_stage_option,
                          flop_count=total_flops,
                          allreduce_groups=allreduce_groups)
    if pipeline_schedule == "1f1b_overlap_friendly":
        emitter_cls = OverlapFriendlyPipelineInstEmitter
        emitter_kwargs["outvar_def_order"] = [
            stage.outvars_def_order() for stage in jax_all_stages
        ]
    else:
        emitter_cls = PipelineInstEmitter
    pipeshard_config = emitter_cls(**emitter_kwargs).compile()

    debug_compilation_time("runtime emitter")
    return pipeshard_config