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
repo = git.Repo("/weka/scratch/tshu2/sye10/habitat/habitat-lab", search_parent_directories=True)
repo.remotes.origin.fetch()
repo.git.checkout("main")
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

class RandomAgent(Agent):
    

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
        self.pickCount = 0
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
            self.dict = {"action": "agent_{}_humanoid_navigate_action".format(self.id), "action_args": {"agent_{}_oracle_nav_lookat_action".format(self.id): object_trans, "agent_{}_mode".format(self.id): 1}}
            self.start = False
            return self.dict

        if self.next_step == "drop":
            self.next_step = "walk"
            self.hold = False
            self.start = True
            return {"action": ("agent_{}_humanoid_drop_action".format(self.id)), "action_args": {}}
        if self.next_step == "pick":
            if self.pickCount < 100:
                self.next_step = "pick"
                self.pickCount += 1
                return {"action": ("agent_{}_humanoid_pick_obj_id_action".format(self.id)), "action_args": {"agent_{}_humanoid_pick_obj_id".format(self.id): self.obj_id}}
            self.next_step = "walk"
            self.hold = True
            self.start = True
            self.pickCount = 0
            return {"action": ("agent_{}_humanoid_pick_obj_id_action".format(self.id)), "action_args": {"agent_{}_humanoid_pick_obj_id".format(self.id): self.obj_id}}
    def reset(self):
        self.next_index = id
        self.next_step = "walk"
        self.hold = False
        self.obj_id = 0
        self.agent_displ = np.inf
        self.agent_rot = np.inf
        self.dict = {}
        self.start = True
        self.pickCount = 0

@registry.register_task_action
class DropAction(ArticulatedAgentAction):
    def step(self, *args, **kwargs):
        self.cur_grasp_mgr.desnap()

action_dict = {
    "humanoid_joint_action": HumanoidJointActionConfig(),
    "humanoid_navigate_action": OracleNavActionConfig(type="OracleNavCoordinateAction", 
                                                      motion_control="human_joints",
                                                      spawn_max_dist_to_obj=1.0),
    "humanoid_pick_obj_id_action": HumanoidPickActionConfig(type="HumanoidPickObjIdAction"),
    "humanoid_drop_action": ActionConfig(type="DropAction")
}


multi_agent_action_dict = {}
for action_name, action_config in action_dict.items():
    for agent_id in range(2):
        multi_agent_action_dict[f"agent_{agent_id}_{action_name}"] = action_config

config = habitat.get_config("/weka/scratch/tshu2/sye10/habitat/example_scenario_2/configuration/environment.yaml")
from habitat.config.read_write import read_write
main_agent_config = AgentConfig()
urdf_path = os.path.join(data_path, 'hab3_bench_assets/humanoids/female_0/female_0.urdf')
main_agent_config.articulated_agent_urdf = urdf_path
main_agent_config.motion_data_path = os.path.join(data_path, "hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl")
main_agent_config.articulated_agent_type = "KinematicHumanoid"


# Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
# We will later talk about why we are giving the sensors these names
main_agent_config.sim_sensors = {
    "third_rgb": ThirdRGBSensorConfig(),
    "head_rgb": HeadRGBSensorConfig(),
}
import copy
second_agent_config = copy.deepcopy(main_agent_config)



agent_dict = {"agent_0": main_agent_config, "agent_1": second_agent_config}
with read_write(config):
    config.habitat.simulator.update({"agents": agent_dict})
    config.habitat.task.actions.update(multi_agent_action_dict)
env = habitat.Env(config=config)
env.reset()
agent0 = RandomAgent("Normal", env, 0)
agent1 = RandomAgent("Hinder", env, 1)
for _ in range (20):
    env.reset()
    agent0 = RandomAgent("Normal", env, 0)
    agent1 = RandomAgent("Hinder", env, 1)
    rom = env.sim.get_rigid_object_manager()

    for obj_id in env.sim.scene_obj_ids:
        print("------------------------")
        print(obj_id, rom.get_object_by_id(obj_id).translation)

    observations = []
    print("start position: ", env.sim.agents_mgr[0].articulated_agent.base_pos)
    print("start position: ", env.sim.agents_mgr[1].articulated_agent.base_pos)
    vis_frames = []
    observations = []
    i = 0
    print("-------------------")
    print(env.current_episode.episode_id)
    print("-------------------")
    while not env.episode_over:
        action_dict = {}
        action_0 = agent0.act()
        action_1 = agent1.act()
        print(action_0)
        print(action_1)
        action_dict["action"] = (action_0["action"], action_1["action"])
        action_dict["action_args"] = {}
        for key in action_0["action_args"].keys():
            action_dict["action_args"][key] = action_0["action_args"][key]
        for key in action_1["action_args"].keys():
            action_dict["action_args"][key] = action_1["action_args"][key]
        print(action_dict)


        observation = env.step(action_dict)
        observations.append(observation)
        info = env.get_metrics()
        frame = observations_to_image(observation, info)
        frame = overlay_frame(frame, info)
        vis_frames.append(frame)


        print("step {} position:".format(i), env.sim.agents_mgr[0].articulated_agent.base_pos)
        print("step {} rotation:".format(i), env.sim.agents_mgr[0].articulated_agent.base_rot)
        print("step {} position:".format(i), env.sim.agents_mgr[1].articulated_agent.base_pos)
        print("step {} rotation:".format(i), env.sim.agents_mgr[0].articulated_agent.base_rot)
        i += 1
    current_episode = env.current_episode
    video_name = f"{os.path.basename(current_episode.scene_id)}_{current_episode.episode_id}"
    images_to_video(vis_frames, "/weka/scratch/tshu2/sye10/habitat/example_scenario_2/slurm_scripts/video_outputs/HSSD-HAB", video_name, fps=6, quality=9)

"""

    vut.make_video(
        observations,
        "agent_0_third_rgb",
        "color",
        "/weka/scratch/tshu2/sye10/habitat/example_scenario_2/slurm_scripts/video_outputs/HSSD-HAB/Agent0_{}".format(current_episode.episode_id),
        open_vid=False,
    )
    vut.make_video(
        observations,
        "agent_1_third_rgb",
        "color",
        "/weka/scratch/tshu2/sye10/habitat/example_scenario_2/slurm_scripts/video_outputs/HSSD-HAB/Agent1_{}".format(current_episode.episode_id),
        open_vid=False,
    )
    """