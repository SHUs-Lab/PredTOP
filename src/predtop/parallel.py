"""
Parallelism setting

Modified based on the reference from:
https://github.com/alpa-projects/alpa/blob/main/alpa/parallel_method.py

"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Union, Any

from jax import linear_util as lu
from jax.core import AbstractValue
from jax.interpreters import pxla
from jax.tree_util import PyTreeDef
import numpy as np

from alpa.device_mesh import (VirtualPhysicalMesh, get_global_virtual_physical_mesh)
from predtop.compile import compile_pipeshard_executable
from alpa.pipeline_parallel.layer_construction import (LayerOption,
                                                       AutoLayerOption,
                                                       ManualLayerOption)
from alpa.pipeline_parallel.stage_construction import (StageOption,
                                                       AutoStageOption,
                                                       UniformStageOption)
from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.shard_parallel.manual_sharding import ManualShardingOption
from alpa.parallel_method import ParallelMethod


class PipeshardParallel(ParallelMethod):
    """
    Use pipeshard parallelism which combines pipeline parallelism and
    shard parallelism.

    Args:
        devices: Specify the devices to use. If it is None, use all the devices
          in the cluster.
        num_micro_batches: The number of micro batches for gradient
          accumulation.
        default_auto_sharding_option: The default options of the auto-sharding
          solver.
        pipeline_schedule: The pipieline schedules.
          Possible choices: {"1f1b", "gpipe", "inference"}
        layer_option: Options of grouping basic operators to layers.
          Possible choices are {"manual", alpa.AutoLayerOption,
                                 alpa.ManualLayerOption}
        stage_option: Options of grouping layers into pipeline stages.
          Possible choices are {"uniform", "auto", alpa.AutoStageOption,
                                 alpa.ManualStageOption}
        stage_input_shardings: Options of input sharding specs for each stage.
          Shape: [num_pipeline_stages, num_input_vars_in_hlo_module].
    """

    def __init__(
            self,
            devices: Optional[VirtualPhysicalMesh] = None,
            num_micro_batches: int = 1,
            default_auto_sharding_option: Optional[AutoShardingOption] = None,
            pipeline_schedule: str = "1f1b",
            layer_option: Optional[Union[LayerOption, str]] = None,
            stage_option: Optional[Union[StageOption, str]] = None,
            stage_input_shardings: Optional[Sequence[Sequence[
                pxla.ShardingSpec]]] = None,
            manual_sharding_option: ManualShardingOption = None):
        self.devices = devices
        self.num_micro_batches = num_micro_batches
        self.as_option = (default_auto_sharding_option or
                          AutoShardingOption(prefer_reduce_scatter=True))
        self.pipeline_schedule = pipeline_schedule
        if layer_option == "manual":
            layer_option = ManualLayerOption()
        self.layer_option = layer_option or AutoLayerOption(layer_num=2)
        if stage_option == "auto":
            stage_option = AutoStageOption(
                submesh_physical_shape_space="power_of_two",
                submesh_logical_shape_space="single_node_model_parallel",
                stage_imbalance_tolerance=np.inf,
                use_hlo_cost_model=False,
                profiling_database_filename=None,
                cached_profile_result=None,
            )
        elif stage_option == "uniform":
            stage_option = UniformStageOption()
        self.stage_option = stage_option or UniformStageOption()
        self.stage_input_shardings = stage_input_shardings
        assert not (stage_input_shardings is not None and
                    manual_sharding_option is not None)
        self.manual_sharding_option = manual_sharding_option

    def compile_executable(
        self,
        fun: lu.WrappedFun,
        in_tree: PyTreeDef,
        out_tree_thunk: Callable[[], PyTreeDef],
        static_argnums: Sequence[int],
        donated_invars: Sequence[bool],
        batch_invars: Sequence[bool],
        *avals: Sequence[AbstractValue],
    ):
        # Resolve the polymorphism in arguments
        if self.devices is None:
            mesh = get_global_virtual_physical_mesh()
            assert mesh is not None, (
                "Please run `alpa.init()` to initialize alpa.")
        else:
            mesh = self.devices

        assert isinstance(mesh, VirtualPhysicalMesh)

        return compile_pipeshard_executable(
            fun, in_tree, out_tree_thunk, static_argnums, donated_invars,
            batch_invars, mesh, self.num_micro_batches, self.pipeline_schedule,
            self.as_option, self.layer_option, self.stage_option, None,
            self.stage_input_shardings, self.manual_sharding_option, *avals)

