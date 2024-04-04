import os
import numpy as np
import git

import habitat
import gym
import habitat.gym


repo = git.Repo("/weka/scratch/tshu2/sye10/habitat/habitat-sim", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
print(f"data_path = {data_path}")

#configure the save path for video output:
output_directory = os.path.join(
    dir_path, "/weka/scratch/tshu2/sye10/habitat/slurm_scripts/outputs"
)
output_path = os.path.join(dir_path, output_directory)
if not os.path.exists(output_path):
    os.mkdir(output_path)

def test_run():

    with gym.make("HabitatRenderPick-v0") as env:
        print("Environment creation successful")
        observations = env.reset()  # noqa: F841

        print("Agent acting inside environment.")
        count_steps = 0
        terminal = False
        while not terminal:
            observations, reward, terminal, info = env.step(
                env.action_space.sample()
            )  # noqa: F841
            count_steps += 1
        print("Episode finished after {} steps.".format(count_steps))


if __name__ == "__main__":
    test_run()
