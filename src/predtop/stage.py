"""
Core implementations for stage construction algorithms.
The algorithm groups layers into pipeline stages.

Modified based on the reference from:
https://github.com/alpa-projects/alpa/blob/main/alpa/pipeline_parallel/stage_construction.py

"""

import logging
from typing import Sequence, Tuple, Dict, Optional
from dataclasses import dataclass

from jax.core import Var
import numpy as np

from alpa.device_mesh import VirtualPhysicalMesh
from alpa.global_env import global_config
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, merge_marked_jaxprs_with_named_call)

from alpa.pipeline_parallel.stage_profiling import (get_compute_cost)
from predtop.profile import (get_compute_cost_pred, get_submesh_models, pred_latency)
from alpa.pipeline_parallel.stage_construction import (
    inference_dp,
    training_dp,
    StageOption,
    get_sliced_virtual_submeshes,
    get_stage_outvars,
    AutoStageOption,
    ManualStageOption,
    get_submesh_choices,
    get_all_submesh_autosharding_config_choices,
)

from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.timer import timers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class PredStageOption:
    # Layer IDs of each forward stage.
    forward_stage_layer_ids: Sequence[Sequence[int]]
    # The physical shapes of submeshes of each stage.
    submesh_physical_shapes: Sequence[Sequence[int]]
    # The logical shapes of submeshes of each stage.
    submesh_logical_shapes: Sequence[Sequence[int]]
    # The auto-sharding options of each stage.
    submesh_autosharding_option_dicts: Sequence[dict]
    
    # The search space of the physical submesh shapes.
    # Possible choices: {"power_of_two", "small_power_of_two", "all"}.
    submesh_physical_shape_space: str = "power_of_two"
    # The search space of the logical mesh shapes.
    # Possible choices: {"same_as_physical", "data_parallel_only",
    #                    "single_node_model_parallel", "all", "manual"}.
    # If "manual", the user needs to specify the logical mesh shape.
    manually_specified_submeshes: Sequence[Tuple[int, int]] = None
    # The search space for the logical mesh shapes.
    # Possible choices: {"all", "single_node_model_parallel",
    #                    "same_as_physical", "data_parallel_only",
    #                    "model_parallel_only"}.
    submesh_logical_shape_space: str = "single_node_model_parallel"
    # Profile only individual layers or composition different layers.
    # Possible choices: {"individual", "composition"}.
    layer_profile_mode: str = "composition"
    # The tolerance of imbalance in the auto-stage construction.
    stage_imbalance_tolerance: float = np.inf
    # Use HLO cost model for computational cost or profile for the cost.
    use_hlo_cost_model: bool = False
    # The filename of profiling result database.
    profiling_database_filename: Optional[str] = None
    # The file name of the cached compute cost.
    cached_profile_result: Optional[str] = None
    

def lat_pred(
        layers: Sequence[JaxPipelineComputation],
        virtual_mesh: VirtualPhysicalMesh, accumulator_mapping: Dict[Var, Var],
        acc_grad_invars: Sequence[Var], acc_grad_outvars: Sequence[Var],
        num_micro_batches: int, batch_size: int,
        jax_apply_layers: Sequence[JaxPipelineComputation],
        apply_grad_global_info: Tuple, pipeline_schedule: str,
        default_as_option: AutoShardingOption, stage_option: StageOption):

    assert len(layers) % 2 == 0
    submesh_choices = get_submesh_choices(
        virtual_mesh.num_hosts, virtual_mesh.num_devices_per_host,
        stage_option.submesh_physical_shape_space,
        stage_option.manually_specified_submeshes)

    autosharding_configs = get_all_submesh_autosharding_config_choices(
        virtual_mesh, submesh_choices,
        stage_option.submesh_logical_shape_space, batch_size)  

    return pred_latency(
        virtual_mesh,
        submesh_choices,
        autosharding_configs,
        layers,
        accumulator_mapping,
        acc_grad_invars,
        acc_grad_outvars,
        jax_apply_layers,
        apply_grad_global_info,
        num_micro_batches,
        default_as_option,
        stage_option)

def get_mesh_models(
        layers: Sequence[JaxPipelineComputation],
        virtual_mesh: VirtualPhysicalMesh, accumulator_mapping: Dict[Var, Var],
        acc_grad_invars: Sequence[Var], acc_grad_outvars: Sequence[Var],
        num_micro_batches: int, batch_size: int,
        jax_apply_layers: Sequence[JaxPipelineComputation],
        apply_grad_global_info: Tuple, pipeline_schedule: str,
        default_as_option: AutoShardingOption, stage_option: StageOption):
    """
    Train the prediction models for each mesh.

    Args:
        layers: All the layers.
        virtual_mesh: The virtual device mesh.
        accumulator_mapping: The donation_mapping for the layers.
        acc_grad_invars: invars of the gradient accumulation layers.
        acc_grad_outvars: outvars of the gradient accumulation layers.
        num_micro_batches: The number of microbatches.
        batch_size: The micro batch size.
        jax_apply_layers: The apply gradient computations corresponding
          to each forward layers.
        pipeline_schedule: The pipeline schedule.
        default_as_option: The default auto-sharding option.
        stage_option: The options controling how to construct stages.
    """
    # Assume each forward layer corresponds to a backward layer
    assert len(layers) % 2 == 0
    submesh_choices = get_submesh_choices(
        virtual_mesh.num_hosts, virtual_mesh.num_devices_per_host,
        stage_option.submesh_physical_shape_space,
        stage_option.manually_specified_submeshes)

    print(stage_option)
    
    # submesh_choices = ((1, 1), )
    autosharding_configs = get_all_submesh_autosharding_config_choices(
        virtual_mesh, submesh_choices,
        stage_option.submesh_logical_shape_space, batch_size)

    return get_submesh_models(
        virtual_mesh, submesh_choices, autosharding_configs, layers,
        accumulator_mapping, acc_grad_invars, acc_grad_outvars,
        jax_apply_layers, apply_grad_global_info, num_micro_batches,
        default_as_option, stage_option)


def cluster_layers_and_slice_mesh_pred(
        layers: Sequence[JaxPipelineComputation],
        virtual_mesh: VirtualPhysicalMesh, accumulator_mapping: Dict[Var, Var],
        acc_grad_invars: Sequence[Var], acc_grad_outvars: Sequence[Var],
        num_micro_batches: int, batch_size: int,
        jax_apply_layers: Sequence[JaxPipelineComputation],
        apply_grad_global_info: Tuple, pipeline_schedule: str,
        default_as_option: AutoShardingOption, stage_option: StageOption,
        mesh_models: []):
    """
    Stage-mesh assignment.

    This function clusters pipeline layers into stages, slice the device
    mesh into multiple submeshes, and assign the stages to the submeshes.
    We first profile the compute cost of layers on different choices
    of submeshes and find the optimal solution with DP.

    Args:
        layers: All the layers.
        virtual_mesh: The virtual device mesh.
        accumulator_mapping: The donation_mapping for the layers.
        acc_grad_invars: invars of the gradient accumulation layers.
        acc_grad_outvars: outvars of the gradient accumulation layers.
        num_micro_batches: The number of microbatches.
        batch_size: The micro batch size.
        jax_apply_layers: The apply gradient computations corresponding
          to each forward layers.
        pipeline_schedule: The pipeline schedule.
        default_as_option: The default auto-sharding option.
        stage_option: The options controling how to construct stages.
    """
    timers("stage-construction").start()

    inference_mode = (pipeline_schedule == "inference")
    num_layers = len(layers) // 2

    if isinstance(stage_option, AutoStageOption):
        submesh_choices = get_submesh_choices(
            virtual_mesh.num_hosts, virtual_mesh.num_devices_per_host,
            stage_option.submesh_physical_shape_space,
            stage_option.manually_specified_submeshes)
        autosharding_configs = get_all_submesh_autosharding_config_choices(
            virtual_mesh, submesh_choices,
            stage_option.submesh_logical_shape_space, batch_size)
        num_autosharding_configs = len(autosharding_configs[0])

        # Use DP to find the optimal solution.
        compute_cost, max_n_succ_stages = get_compute_cost_pred(
            virtual_mesh, submesh_choices, autosharding_configs, layers,
            accumulator_mapping, acc_grad_invars, acc_grad_outvars,
            jax_apply_layers, apply_grad_global_info, num_micro_batches,
            default_as_option, stage_option, inference_mode, mesh_models)
            

        if inference_mode:
            _, solution = inference_dp(num_layers, virtual_mesh.num_devices,
                                        submesh_choices,
                                        num_autosharding_configs, compute_cost)
        else:
            _, solution = training_dp(num_layers, virtual_mesh.num_devices,
                                        num_micro_batches, submesh_choices,
                                        num_autosharding_configs, compute_cost,
                                        max_n_succ_stages)

        assert solution is not None, "no solution in auto stage construction."

        # Parse solution
        forward_stage_layer_ids = [
            list(range(start_id, end_id))
            for (start_id, end_id), _, _ in solution
        ]
        submesh_shapes = [
            submesh_choices[submesh_id] for _, submesh_id, _ in solution
        ]
        selected_autosharding_configs = [
            autosharding_configs[submesh_id][autosharding_config_id]
            for _, submesh_id, autosharding_config_id in solution
        ]
        logical_mesh_shapes = [
            mesh.shape for mesh, _ in selected_autosharding_configs
        ]
        autosharding_option_dicts = [
            option_dict for _, option_dict in selected_autosharding_configs
        ]


        # TODO: test manual plans
        # forward_stage_layer_ids = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]
        # submesh_shapes = [(1, 2), (1, 2)]
        # logical_mesh_shapes = [(2, 1), (2, 1)]
        # autosharding_option_dicts = [{}, {}]

        # Print and store the results
        print("Result forward_stage_layer_ids:", forward_stage_layer_ids)
        print("Result mesh_shapes:", submesh_shapes)
        print("Result logical_mesh_shapes:", logical_mesh_shapes)
        print("Result autosharding_option_dicts:", autosharding_option_dicts)
        global last_forward_stage_layer_ids, last_submesh_shapes
        global last_logical_mesh_shapes, last_autosharding_option_dicts
        last_forward_stage_layer_ids = forward_stage_layer_ids
        last_submesh_shapes = submesh_shapes
        last_logical_mesh_shapes = logical_mesh_shapes
        last_autosharding_option_dicts = autosharding_option_dicts

    elif isinstance(stage_option, PredStageOption):
        print('in')
        print(stage_option)
        # Check forward_stage_layer_ids is a partition of range(num_layers)
        forward_stage_layer_ids = stage_option.forward_stage_layer_ids
        last_layer_id = 0
        for stage_layer_ids in forward_stage_layer_ids:
            for layer_id in stage_layer_ids:
                assert layer_id == last_layer_id
                last_layer_id += 1
        assert last_layer_id == num_layers, (
            f"{last_layer_id} layers in stage option, but {num_layers} marked")
        submesh_shapes = stage_option.submesh_physical_shapes
        logical_mesh_shapes = (stage_option.submesh_logical_shapes or
                               submesh_shapes)
        autosharding_option_dicts = (
            stage_option.submesh_autosharding_option_dicts)


    sliced_meshes = get_sliced_virtual_submeshes(virtual_mesh,
                                                     submesh_shapes)

    num_forward_stages = len(forward_stage_layer_ids)

    backward_stage_layer_ids = [[
        2 * num_layers - 1 - i for i in reversed(layer_ids)
    ] for layer_ids in reversed(forward_stage_layer_ids)]
    stage_layer_ids = forward_stage_layer_ids + backward_stage_layer_ids
    stage_to_mesh = list(range(num_forward_stages)) + list(
        reversed(range(num_forward_stages)))

    stage_outvars = get_stage_outvars(layers, stage_layer_ids, acc_grad_outvars)
    merged_stages = []
    for stage_id, layer_ids in enumerate(stage_layer_ids):
        if len(layer_ids) == 1:
            merged_stages.append(layers[layer_ids[0]])
            continue

        stage_layer_jaxprs = [layers[i].closed_jaxpr() for i in layer_ids]
        stage_name = str(stage_id)
        merged_stage_jaxpr = merge_marked_jaxprs_with_named_call(
            stage_layer_jaxprs,
            stage_outvars[stage_id],
            accumulator_mapping,
            stage_name,
            wrap_with_marker=True)
        merged_stage = JaxPipelineComputation.from_closed_jaxpr(
            stage_name, merged_stage_jaxpr)
        merged_stages.append(merged_stage)
    stages = merged_stages

    # Check the validity of logical mesh shapes
    assert len(logical_mesh_shapes) == len(sliced_meshes)
    for logical_mesh_shape, submesh in zip(logical_mesh_shapes, sliced_meshes):
        assert np.prod(logical_mesh_shape) == submesh.num_devices

    if autosharding_option_dicts is not None:
        assert len(autosharding_option_dicts) == len(sliced_meshes)
    else:
        autosharding_option_dicts = [{}] * len(sliced_meshes)

    manual_stage_option = ManualStageOption(
        forward_stage_layer_ids, tuple(x.shape for x in sliced_meshes),
        logical_mesh_shapes, autosharding_option_dicts)

    timers("stage-construction").stop()
    return stages, stage_to_mesh, sliced_meshes, manual_stage_option


def cluster_layers_and_slice_mesh(
        layers: Sequence[JaxPipelineComputation],
        virtual_mesh: VirtualPhysicalMesh, accumulator_mapping: Dict[Var, Var],
        acc_grad_invars: Sequence[Var], acc_grad_outvars: Sequence[Var],
        num_micro_batches: int, batch_size: int,
        jax_apply_layers: Sequence[JaxPipelineComputation],
        apply_grad_global_info: Tuple, pipeline_schedule: str,
        default_as_option: AutoShardingOption, stage_option: StageOption):
    """
    Stage-mesh assignment.

    This function clusters pipeline layers into stages, slice the device
    mesh into multiple submeshes, and assign the stages to the submeshes.
    We first profile the compute cost of layers on different choices
    of submeshes and find the optimal solution with DP.

    Args:
        layers: All the layers.
        virtual_mesh: The virtual device mesh.
        accumulator_mapping: The donation_mapping for the layers.
        acc_grad_invars: invars of the gradient accumulation layers.
        acc_grad_outvars: outvars of the gradient accumulation layers.
        num_micro_batches: The number of microbatches.
        batch_size: The micro batch size.
        jax_apply_layers: The apply gradient computations corresponding
          to each forward layers.
        pipeline_schedule: The pipeline schedule.
        default_as_option: The default auto-sharding option.
        stage_option: The options controling how to construct stages.
    """
    timers("stage-construction").start()

    inference_mode = (pipeline_schedule == "inference")
    if virtual_mesh.launched_physical_mesh_group is None:
        given_mesh = False
    else:
        given_mesh = True

    if inference_mode:
        num_layers = len(layers)
    else:
        # Assume each forward layer corresponds to a backward layer
        assert len(layers) % 2 == 0
        num_layers = len(layers) // 2

    if isinstance(stage_option, AutoStageOption):
        if given_mesh:
            # TODO(zhuohan): Implement the auto slicing with given mesh.
            raise NotImplementedError("automatically slicing layers with "
                                      "existing physical meshes is not"
                                      "supported yet.")

        submesh_choices = get_submesh_choices(
            virtual_mesh.num_hosts, virtual_mesh.num_devices_per_host,
            stage_option.submesh_physical_shape_space,
            stage_option.manually_specified_submeshes)
        autosharding_configs = get_all_submesh_autosharding_config_choices(
            virtual_mesh, submesh_choices,
            stage_option.submesh_logical_shape_space, batch_size)
        num_autosharding_configs = len(autosharding_configs[0])

        # Use DP to find the optimal solution.
        compute_cost, max_n_succ_stages = get_compute_cost(
            virtual_mesh, submesh_choices, autosharding_configs, layers,
            accumulator_mapping, acc_grad_invars, acc_grad_outvars,
            jax_apply_layers, apply_grad_global_info, num_micro_batches,
            default_as_option, stage_option, inference_mode)
        if inference_mode:
            _, solution = inference_dp(num_layers, virtual_mesh.num_devices,
                                       submesh_choices,
                                       num_autosharding_configs, compute_cost)
        else:
            _, solution = training_dp(num_layers, virtual_mesh.num_devices,
                                      num_micro_batches, submesh_choices,
                                      num_autosharding_configs, compute_cost,
                                      max_n_succ_stages)

        assert solution is not None, "no solution in auto stage construction."

        # Parse solution
        forward_stage_layer_ids = [
            list(range(start_id, end_id))
            for (start_id, end_id), _, _ in solution
        ]
        submesh_shapes = [
            submesh_choices[submesh_id] for _, submesh_id, _ in solution
        ]
        selected_autosharding_configs = [
            autosharding_configs[submesh_id][autosharding_config_id]
            for _, submesh_id, autosharding_config_id in solution
        ]
        logical_mesh_shapes = [
            mesh.shape for mesh, _ in selected_autosharding_configs
        ]
        autosharding_option_dicts = [
            option_dict for _, option_dict in selected_autosharding_configs
        ]

        # Print and store the results
        print("Result forward_stage_layer_ids:", forward_stage_layer_ids)
        print("Result mesh_shapes:", submesh_shapes)
        print("Result logical_mesh_shapes:", logical_mesh_shapes)
        print("Result autosharding_option_dicts:", autosharding_option_dicts)
        global last_forward_stage_layer_ids, last_submesh_shapes
        global last_logical_mesh_shapes, last_autosharding_option_dicts
        last_forward_stage_layer_ids = forward_stage_layer_ids
        last_submesh_shapes = submesh_shapes
        last_logical_mesh_shapes = logical_mesh_shapes
        last_autosharding_option_dicts = autosharding_option_dicts
    elif isinstance(stage_option, ManualStageOption):
        # Check forward_stage_layer_ids is a partition of range(num_layers)
        forward_stage_layer_ids = stage_option.forward_stage_layer_ids
        last_layer_id = 0
        for stage_layer_ids in forward_stage_layer_ids:
            for layer_id in stage_layer_ids:
                assert layer_id == last_layer_id
                last_layer_id += 1
        assert last_layer_id == num_layers, (
            f"{last_layer_id} layers in stage option, but {num_layers} marked")
        submesh_shapes = stage_option.submesh_physical_shapes
        logical_mesh_shapes = (stage_option.submesh_logical_shapes or
                               submesh_shapes)
        autosharding_option_dicts = (
            stage_option.submesh_autosharding_option_dicts)
    else:
        raise ValueError(f"Invalid pipeline stage option: {stage_option}")

    if given_mesh:
        sliced_meshes = [
            mesh.get_virtual_physical_mesh()
            for mesh in virtual_mesh.launched_physical_mesh_group
        ]
    else:
        sliced_meshes = get_sliced_virtual_submeshes(virtual_mesh,
                                                     submesh_shapes)

    num_forward_stages = len(forward_stage_layer_ids)

    if inference_mode:
        stage_layer_ids = forward_stage_layer_ids
        stage_to_mesh = list(range(num_forward_stages))
    else:
        backward_stage_layer_ids = [[
            2 * num_layers - 1 - i for i in reversed(layer_ids)
        ] for layer_ids in reversed(forward_stage_layer_ids)]
        stage_layer_ids = forward_stage_layer_ids + backward_stage_layer_ids
        stage_to_mesh = list(range(num_forward_stages)) + list(
            reversed(range(num_forward_stages)))

    stage_outvars = get_stage_outvars(layers, stage_layer_ids, acc_grad_outvars)
    merged_stages = []
    for stage_id, layer_ids in enumerate(stage_layer_ids):
        if len(layer_ids) == 1:
            merged_stages.append(layers[layer_ids[0]])
            continue

        stage_layer_jaxprs = [layers[i].closed_jaxpr() for i in layer_ids]
        stage_name = str(stage_id)
        merged_stage_jaxpr = merge_marked_jaxprs_with_named_call(
            stage_layer_jaxprs,
            stage_outvars[stage_id],
            accumulator_mapping,
            stage_name,
            wrap_with_marker=True)
        merged_stage = JaxPipelineComputation.from_closed_jaxpr(
            stage_name, merged_stage_jaxpr)
        merged_stages.append(merged_stage)
    stages = merged_stages

    # Check the validity of logical mesh shapes
    assert len(logical_mesh_shapes) == len(sliced_meshes)
    for logical_mesh_shape, submesh in zip(logical_mesh_shapes, sliced_meshes):
        assert np.prod(logical_mesh_shape) == submesh.num_devices

    if autosharding_option_dicts is not None:
        assert len(autosharding_option_dicts) == len(sliced_meshes)
    else:
        autosharding_option_dicts = [{}] * len(sliced_meshes)

    manual_stage_option = ManualStageOption(
        forward_stage_layer_ids, tuple(x.shape for x in sliced_meshes),
        logical_mesh_shapes, autosharding_option_dicts)

    timers("stage-construction").stop()
    return stages, stage_to_mesh, sliced_meshes, manual_stage_option
