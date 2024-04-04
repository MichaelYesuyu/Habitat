import gzip
import json
import os
import sys
from typing import Any, Dict, List, Optional, Type

import attr
import cv2
import git
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image

import habitat
import habitat_sim
#from habitat.config import Config
from habitat.core.registry import registry
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

repo = git.Repo("/weka/scratch/tshu2/sye10/habitat/habitat-sim", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
output_directory = "/weka/scratch/tshu2/sye10/habitat/outputs" 
output_path = os.path.join(dir_path, output_directory)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument(
        "--no-make-video", dest="make_video", action="store_false"
    )
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = False
    display = False
    make_video = args.make_video
else:
    show_video = False
    make_video = False
    display = False

if make_video and not os.path.exists(output_path):
    os.makedirs(output_path)

# @title Util functions to visualize observations
# @markdown - `make_video_cv2`: Renders a video from a list of observations
# @markdown - `simulate`: Runs simulation for a given amount of time at 60Hz
# @markdown - `simulate_and_make_vid` Runs simulation and creates video


def make_video_cv2(
    observations, cross_hair=None, prefix="", open_vid=True, fps=60
):
    sensor_keys = list(observations[0])
    videodims = observations[0][sensor_keys[0]].shape
    videodims = (videodims[1], videodims[0])  # flip to w,h order
    print(videodims)
    video_file = output_path + prefix + ".mp4"
    print("Encoding the video: %s " % video_file)
    writer = vut.get_fast_video_writer(video_file, fps=fps)
    for ob in observations:
        # If in RGB/RGBA format, remove the alpha channel
        rgb_im_1st_person = cv2.cvtColor(ob["rgb"], cv2.COLOR_RGBA2RGB)
        if cross_hair is not None:
            rgb_im_1st_person[
                cross_hair[0] - 2 : cross_hair[0] + 2,
                cross_hair[1] - 2 : cross_hair[1] + 2,
            ] = [255, 0, 0]

        if rgb_im_1st_person.shape[:2] != videodims:
            rgb_im_1st_person = cv2.resize(
                rgb_im_1st_person, videodims, interpolation=cv2.INTER_AREA
            )
        # write the 1st person observation to video
        writer.append_data(rgb_im_1st_person)
    writer.close()

    if open_vid:
        print("Displaying video")
        vut.display_video(video_file)


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations


# convenience wrapper for simulate and make_video_cv2
def simulate_and_make_vid(sim, crosshair, prefix, dt=1.0, open_vid=True):
    observations = simulate(sim, dt)
    make_video_cv2(observations, crosshair, prefix=prefix, open_vid=open_vid)


def display_sample(
    rgb_obs,
    semantic_obs=np.array([]),
    depth_obs=np.array([]),
    key_points=None,  # noqa: B006
):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray(
            (depth_obs / 10 * 255).astype(np.uint8), mode="L"
        )
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        # plot points on images
        if key_points is not None:
            for point in key_points:
                plt.plot(
                    point[0], point[1], marker="o", markersize=10, alpha=0.8
                )
        plt.imshow(data)

    plt.show(block=False)

# @title Setup simulator configuration
# @markdown We'll start with setting up simulator with the following configurations
# @markdown - The simulator will render both RGB, Depth observations of 256x256 resolution.
# @markdown - The actions available will be `move_forward`, `turn_left`, `turn_right`.


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.default_agent_id = settings["default_agent_id"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.physics_config_file = settings["physics_config_file"]

    # Note: all sensors must have the same resolution
    sensors = {
        "rgb": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

test_scene = os.path.join(
    data_path, "scene_datasets/coda/coda.glb"
)


settings = {
    "max_frames": 10,
    "width": 256,
    "height": 256,
    "scene": test_scene,
    "default_agent_id": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "rgb": True,  # RGB sensor
    "depth": True,  # Depth sensor
    "seed": 1,
    "enable_physics": True,
    "physics_config_file": "data/default.phys_scene_config.json",
    "silent": False,
    "compute_shortest_path": False,
    "compute_action_shortest_path": False,
    "save_png": True,
}

cfg = make_cfg(settings)

# @title Spawn the agent at a pre-defined location


def init_agent(sim):
    agent_pos = np.array([-0.15776923, 0.18244143, 0.2988735])

    # Place the agent
    sim.agents[0].scene_node.translation = agent_pos
    agent_orientation_y = -40
    sim.agents[0].scene_node.rotation = mn.Quaternion.rotation(
        mn.Deg(agent_orientation_y), mn.Vector3(0, 1.0, 0)
    )


cfg.sim_cfg.default_agent_id = 0
with habitat_sim.Simulator(cfg) as sim:
    init_agent(sim)
    if make_video:
        # Visualize the agent's initial position
        simulate_and_make_vid(
            sim, None, "sim-init", dt=1.0, open_vid=show_video
        )

# @title Set the object's initial and final position
# @markdown Defines two utility functions:
# @markdown - `remove_all_objects`: This will remove all objects from the scene
# @markdown - `set_object_in_front_of_agent`: This will add an object in the scene in front of the agent at the specified distance.

# @markdown Here we add a chair *3.0m* away from the agent and the task is to place the agent at the desired final position which is *7.0m* in front of the agent.


def remove_all_objects(sim):
    for id in sim.get_existing_object_ids():
        sim.remove_object(id)


def set_object_in_front_of_agent(sim, obj_id, z_offset=-1.5):
    r"""
    Adds an object in front of the agent at some distance.
    """
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    obj_translation = agent_transform.transform_point(
        np.array([0, 0, z_offset])
    )
    sim.set_translation(obj_translation, obj_id)

    obj_node = sim.get_object_scene_node(obj_id)
    xform_bb = habitat_sim.geo.get_transformed_bb(
        obj_node.cumulative_bb, obj_node.transformation
    )

    # also account for collision margin of the scene
    scene_collision_margin = 0.04
    y_translation = mn.Vector3(
        0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
    )
    sim.set_translation(y_translation + sim.get_translation(obj_id), obj_id)


def init_objects(sim):
    # Manager of Object Attributes Templates
    obj_attr_mgr = sim.get_object_template_manager()

    # Add a chair into the scene.
    obj_path = "test_assets/objects/chair"
    chair_template_id = obj_attr_mgr.load_object_configs(
        str(os.path.join(data_path, obj_path))
    )[0]
    chair_attr = obj_attr_mgr.get_template_by_ID(chair_template_id)
    obj_attr_mgr.register_template(chair_attr)

    # Object's initial position 3m away from the agent.
    object_id = sim.add_object_by_handle(chair_attr.handle)
    set_object_in_front_of_agent(sim, object_id, -3.0)
    sim.set_object_motion_type(
        habitat_sim.physics.MotionType.STATIC, object_id
    )

    # Object's final position 7m away from the agent
    goal_id = sim.add_object_by_handle(chair_attr.handle)
    set_object_in_front_of_agent(sim, goal_id, -7.0)
    sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, goal_id)

    return object_id, goal_id


with habitat_sim.Simulator(cfg) as sim:
    init_agent(sim)
    init_objects(sim)

    # Visualize the scene after the chair is added into the scene.
    if make_video:
        simulate_and_make_vid(
            sim, None, "object-init", dt=1.0, open_vid=show_video
        )

# @title Create a new dataset
# @markdown Utility functions to define and save the dataset for the rearrangement task


def get_rotation(sim, object_id):
    quat = sim.get_rotation(object_id)
    return np.array(quat.vector).tolist() + [quat.scalar]


def init_episode_dict(episode_id, scene_id, agent_pos, agent_rot):
    episode_dict = {
        "episode_id": episode_id,
        "scene_id": "data/scene_datasets/coda/coda.glb",
        "start_position": agent_pos,
        "start_rotation": agent_rot,
        "info": {},
    }
    return episode_dict


def add_object_details(sim, episode_dict, id, object_template, object_id):
    object_template = {
        "object_id": id,
        "object_template": object_template,
        "position": np.array(sim.get_translation(object_id)).tolist(),
        "rotation": get_rotation(sim, object_id),
    }
    episode_dict["objects"] = object_template
    return episode_dict


def add_goal_details(sim, episode_dict, object_id):
    goal_template = {
        "position": np.array(sim.get_translation(object_id)).tolist(),
        "rotation": get_rotation(sim, object_id),
    }
    episode_dict["goals"] = goal_template
    return episode_dict

