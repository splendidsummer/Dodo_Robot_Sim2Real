from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    return asset.data.root_link_pos_w[:, 2] < minimum_height 


def root_height_above_maximum(
    env: ManagerBasedRLEnv, maximum_height: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is above the maximum height.

    Note:
        This is currently only supported for flat terrains, i.e. the maximum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] > maximum_height