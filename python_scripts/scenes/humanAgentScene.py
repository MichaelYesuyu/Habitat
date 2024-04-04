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

action_dict = {
    "humanoid_joint_action": HumanoidJointActionConfig(),
    "humanoid_navigate_action": OracleNavActionConfig(type="OracleNavCoordinateAction", 
                                                      motion_control="human_joints",
                                                      spawn_max_dist_to_obj=1.0),
    "humanoid_pick_obj_id_action": HumanoidPickActionConfig(type="HumanoidPickObjIdAction"),
    "humanoid_drop_action": ActionConfig(type="DropAction")
}
def make_sim_cfg(agent_dict):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    
    # Set up an example scene
    sim_cfg.scene = "data/hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json"
    sim_cfg.scene_dataset = "data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json"
    sim_cfg.additional_object_paths = ['data/objects/ycb/configs/']

    
    cfg = OmegaConf.create(sim_cfg)

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg

def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()
    dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/hab3_bench_assets/episode_datasets/small_large.json.gz")
    
    
    hab_cfg = HabitatConfig()
    hab_cfg.environment = env_cfg
    hab_cfg.task = task_cfg
    hab_cfg.dataset = dataset_cfg
    hab_cfg.simulator = sim_cfg
    hab_cfg.simulator.seed = hab_cfg.seed

    return hab_cfg

def init_rearrange_env(agent_dict, action_dict):
    hab_cfg = make_hab_cfg(agent_dict, action_dict)
    res_cfg = OmegaConf.create(hab_cfg)
    return Env(res_cfg)

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
agent_dict = {"agent_0": main_agent_config}
env = init_rearrange_env(agent_dict, action_dict)
env.reset()
rom = env.sim.get_rigid_object_manager()
# env.sim.articulated_agent.base_pos = init_pos
# As before, we get a navigation point next to an object id

obj_id = env.sim.scene_obj_ids[0]
first_object = rom.get_object_by_id(obj_id)

object_trans = first_object.translation

observations = []
delta = 2.0

object_agent_vec = env.sim.articulated_agent.base_pos - object_trans
object_agent_vec.y = 0
dist_agent_object = object_agent_vec.length()
# Walk towards the object

agent_displ = np.inf
agent_rot = np.inf
prev_rot = env.sim.articulated_agent.base_rot
prev_pos = env.sim.articulated_agent.base_pos
while agent_displ > 1e-9 or agent_rot > 1e-9:
    prev_rot = env.sim.articulated_agent.base_rot
    prev_pos = env.sim.articulated_agent.base_pos
    action_dict = {
        "action": ("humanoid_navigate_action"), 
        "action_args": {
              "oracle_nav_lookat_action": object_trans,
              "mode": 1
          }
    }
    observations.append(env.step(action_dict))
    
    cur_rot = env.sim.articulated_agent.base_rot
    cur_pos = env.sim.articulated_agent.base_pos
    agent_displ = (cur_pos - prev_pos).length()
    agent_rot = np.abs(cur_rot - prev_rot)
    
# Wait
for _ in range(20):
    action_dict = {"action": (), "action_args": {}}
    observations.append(env.step(action_dict))

# Pick object
observations.append(env.step(action_dict))
for _ in range(100):
    
    action_dict = {"action": ("humanoid_pick_obj_id_action"), "action_args": {"humanoid_pick_obj_id": obj_id}}
    observations.append(env.step(action_dict))

for _ in range(20):
    action_dict = {"action": (), "action_args": {}}
    observations.append(env.step(action_dict))

    
vut.make_video(
    observations,
    "third_rgb",
    "color",
    "/weka/scratch/tshu2/sye10/habitat/video_outputs/HumanAction",
    open_vid=False,
)
env.reset()
observations = []
motion_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl" 
# We define here humanoid controller
humanoid_controller = HumanoidRearrangeController(motion_path)
humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
print(env.sim.articulated_agent.base_pos)
for _ in range(100):
    # This computes a pose that moves the agent to relative_position
    relative_position = env.sim.articulated_agent.base_pos + mn.Vector3(0,0,1)
    humanoid_controller.calculate_walk_pose(relative_position)
    
    # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }
    observations.append(env.step(action_dict))


    
vut.make_video(
    observations,
    "third_rgb",
    "color",
    "/weka/scratch/tshu2/sye10/habitat/video_outputs/HumanController",
    open_vid=False,
)

env.reset()
humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
observations = []
print(env.sim.articulated_agent.base_pos)

# Get the hand pose
offset =  env.sim.articulated_agent.base_transformation.transform_vector(mn.Vector3(0, 0.3, 0))
hand_pose = env.sim.articulated_agent.ee_transform(0).translation + offset
for _ in range(300):
    # This computes a pose that moves the agent to relative_position
    hand_pose = hand_pose + mn.Vector3((np.random.rand(3) - 0.5) * 0.1)
    humanoid_controller.calculate_reach_pose(hand_pose, index_hand=0)
    
    # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }
    observations.append(env.step(action_dict))
    
vut.make_video(
    observations,
    "third_rgb",
    "color",
    "/weka/scratch/tshu2/sye10/habitat/video_outputs/HumanControllerPick",
    open_vid=False,
)