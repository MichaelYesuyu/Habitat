import habitat
import matplotlib.pyplot as plt
import numpy as np
from habitat.core.agent import Agent

import habitat_sim
import magnum as mn
import warnings
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
from habitat_sim.utils import viz_utils as vut
from omegaconf import DictConfig
import numpy as np
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig, IteratorOptionsConfig, TopDownMapMeasurementConfig, CollisionsMeasurementConfig
from habitat.config.default import get_agent_config
import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from omegaconf import OmegaConf
from PIL import Image
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
import random

import git, os
repo = git.Repo("/weka/scratch/tshu2/sye10/habitat/habitat-sim", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
os.chdir(dir_path)
from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from habitat.config.default_structured_configs import ArmActionConfig, BaseVelocityActionConfig, OracleNavActionConfig, ActionConfig, FogOfWarConfig
from habitat.core.env import Env
from habitat.tasks.rearrange.actions.articulated_agent_action import ArticulatedAgentAction
from habitat.core.registry import registry
from gym import spaces
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)
from habitat.config.default_structured_configs import HumanoidJointActionConfig, HumanoidPickActionConfig

class HumanAgent(Agent):

    def __init__(self, name: str, env: habitat.Env, id:int):
        self.name = name
        self.env = env
        self.id = id
        self.next_step = "walk"
        self.hold = False
        self.close = False
        self.obj_id = 0
        self.next_index = id
        self.agent_displ = np.inf
        self.agent_rot = np.inf
        self.prev_pos = env.sim.agents_mgr[self.id].articulated_agent.base_pos
        self.prev_rot = env.sim.agents_mgr[self.id].articulated_agent.base_rot
        self.dict = {}
        self.start = True
    def act(self, observations = True):


        cur_pos = env.sim.agents_mgr[self.id].articulated_agent.base_pos
        cur_rot = env.sim.agents_mgr[self.id].articulated_agent.base_rot

        if cur_pos == self.prev_pos and cur_rot == self.prev_rot and self.start == False:
            if self.hold:
                self.next_step = "drop"
            else:
                self.next_step = "pick"

        if cur_pos != self.prev_pos or cur_rot != self.prev_rot:
            self.agent_displ = (cur_pos - self.prev_pos).length()
            self.agent_rot = np.abs(cur_rot - self.prev_rot)
            if self.agent_displ > 1e-9 or self.agent_rot > 1e-9:
                self.prev_pos = cur_pos
                self.prev_rot = cur_rot
                return self.dict
            else:
                print("close enough")
                self.prev_pos = cur_pos
                self.prev_rot = cur_rot
                if self.hold:
                    self.next_step = "drop"
                else:
                    self.next_step = "pick"

        if self.next_step == "walk":
            length = len(self.env.sim.scene_obj_ids)
            index = self.next_index
            self.next_index = (self.next_index + 2) % length
            if self.env.sim.scene_obj_ids[self.next_index] == 132 or self.env.sim.scene_obj_ids[self.next_index] == 135 or self.env.sim.scene_obj_ids[self.next_index] == 139:
                self.next_index = (self.next_index + 2) % length
            rom = self.env.sim.get_rigid_object_manager()
            first_object = rom.get_object_by_id(self.env.sim.scene_obj_ids[index])
            object_trans = first_object.translation
            self.obj_id = self.env.sim.scene_obj_ids[index]
            self.dict = {"action": "agent_{}_oracle_coord_action".format(self.id), "action_args": {"agent_{}_oracle_nav_lookat_action".format(self.id): object_trans, "agent_{}_mode".format(self.id): 1}}
            self.start = False
            return self.dict

        if self.next_step == "drop":
            self.next_step = "walk"
            self.hold = False
            self.start = True
            return {"action": ("agent_{}_drop_action".format(self.id)), "action_args": {}}
        if self.next_step == "pick":
            self.next_step = "walk"
            self.hold = True
            self.start = True
            return {"action": ("agent_{}_pick_object_id_action".format(self.id)), "action_args": {"agent_{}_pick_object_id".format(self.id): self.obj_id}}
    def reset(self):
        self.next_index = id
        self.next_step = "walk"
        self.hold = False
        self.obj_id = 0
        self.agent_displ = np.inf
        self.agent_rot = np.inf

@registry.register_task_action
class DropAction(ArticulatedAgentAction):
    def step(self, *args, **kwargs):
        self.cur_grasp_mgr.desnap()
