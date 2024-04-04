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
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from habitat.config.default import get_agent_config
import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from omegaconf import OmegaConf

import git, os
repo = git.Repo("/weka/scratch/tshu2/sye10/habitat/habitat-lab", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
os.chdir(dir_path)

# Define the agent configuration
main_agent_config = AgentConfig()
urdf_path = os.path.join(data_path, "robots/hab_fetch/robots/hab_fetch.urdf")
main_agent_config.articulated_agent_urdf = urdf_path
main_agent_config.articulated_agent_type = "FetchRobot"
#urdf_path = os.path.join(data_path, "hab3_bench_assets/humanoids/female_0/female_0.urdf")
#main_agent_config.articulated_agent_urdf = urdf_path
#main_agent_config.motion_data_path = os.path.join(data_path, "hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl")
#main_agent_config.articulated_agent_type = "KinematicHumanoid"

# Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
# We will later talk about why we are giving the sensors these names
main_agent_config.sim_sensors = {
    "third_rgb": ThirdRGBSensorConfig(),
    "head_rgb": HeadRGBSensorConfig(),
}

# We create a dictionary with names of agents and their corresponding agent configuration
agent_dict = {"main_agent": main_agent_config}

def make_sim_cfg(agent_dict):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    
    # Set up an example scene
    sim_cfg.scene = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json")
    sim_cfg.scene_dataset = os.path.join(data_path, "hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json")
    sim_cfg.additional_object_paths = [os.path.join(data_path, 'objects/ycb/configs/')]

    
    cfg = OmegaConf.create(sim_cfg)

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg


def init_rearrange_sim(agent_dict):
    # Start the scene config
    sim_cfg = make_sim_cfg(agent_dict)    
    cfg = OmegaConf.create(sim_cfg)
    
    # Create the scene
    sim = RearrangeSim(cfg)

    # This is needed to initialize the agents
    sim.agents_mgr.on_new_scene()

    # For this tutorial, we will also add an extra camera that will be used for third person recording.
    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.uuid = "scene_camera_rgb"

    sim.add_sensor(camera_sensor_spec, 0)

    return sim

sim = init_rearrange_sim(agent_dict)

init_pos = mn.Vector3(-5.5,0,-1.5)
art_agent = sim.articulated_agent
# We will see later about this
art_agent.sim_obj.motion_type = MotionType.KINEMATIC
print("Current agent position:", art_agent.base_pos)
art_agent.base_pos = init_pos 
print("New agent position:", art_agent.base_pos)
# We take a step to update agent position
_ = sim.step({})

observations = sim.get_sensor_observations()
print(observations.keys())
#print("Observations: \n")
#print(observations)

_, ax = plt.subplots(1,len(observations.keys()))

for ind, name in enumerate(observations.keys()):
    ax[ind].imshow(observations[name])
    ax[ind].set_axis_off()
    ax[ind].set_title(name)

print(art_agent.params.cameras.keys())

observations = []
num_iter = 100
pos_delta = mn.Vector3(0.02,0,0)
rot_delta = np.pi / (8 * num_iter)
art_agent.base_pos = init_pos

sim.reset()
#set_fixed_camera(sim)
for _ in range(num_iter):
    # TODO: this actually seems to give issues...
    art_agent.base_pos = art_agent.base_pos + pos_delta
    art_agent.base_rot = art_agent.base_rot + rot_delta
    sim.step({})
    observations.append(sim.get_sensor_observations())
"""
vut.make_video(
    observations,
    "third_rgb",
    "color",
    "/weka/scratch/tshu2/sye10/habitat/example_scenario_2/slurm_scripts/video_outputs/camera_test.mp4",
    open_vid=False,
)
"""


from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
import gzip
import json

# Define the agent configuration
episode_file = os.path.join(data_path, "hab3_bench_assets/episode_datasets/small_large.json.gz")
sim = init_rearrange_sim(agent_dict)
# Load the dataset
with gzip.open(episode_file, "rt") as f: 
    episode_files = json.loads(f.read())

# Get the first episode
episode = episode_files["episodes"][0]
rearrange_episode = RearrangeEpisode(**episode)

art_agent = sim.articulated_agent
art_agent._fixed_base = True
sim.agents_mgr.on_new_scene()


sim.reconfigure(sim.habitat_config, ep_info=rearrange_episode)
sim.reset()

art_agent.sim_obj.motion_type = MotionType.KINEMATIC
sim.articulated_agent.base_pos =  init_pos 
_ = sim.step({})

aom = sim.get_articulated_object_manager()
rom = sim.get_rigid_object_manager()

# We can query the articulated and rigid objects

print("List of articulated objects:")
for handle, ao in aom.get_objects_by_handle_substring().items():
    print(handle, "id", aom.get_object_id_by_handle(handle))

print("\nList of rigid objects:")
obj_ids = []
for handle, ro in rom.get_objects_by_handle_substring().items():
    print(handle, "id", ro.object_id, ro.awake)
    obj_ids.append(ro.object_id)

sim.reset()
art_agent.sim_obj.motion_type = MotionType.KINEMATIC
obj_id = sim.scene_obj_ids[0]
first_object = rom.get_object_by_id(obj_id)

object_trans = first_object.translation
print(first_object.handle, "is at", object_trans)

print("Reached here\n")

sim.reset()
object_ids = sim.scene_obj_ids

for obj_id in object_ids:
    scene_object = rom.get_object_by_id(obj_id)
    object_trans = scene_object.translation
    object_rot = scene_object.rotation
    print(f"{scene_object.handle} is located at {object_trans}")
    print(f"{scene_object.handle} rotation is {object_rot}\n")

