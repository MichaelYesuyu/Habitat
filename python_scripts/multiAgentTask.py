import habitat
from habitat.config.default import get_config

import git
import os

repo = git.Repo("/weka/scratch/tshu2/sye10/habitat/habitat-sim", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
print(f"data_path = {data_path}")
# @markdown Optionally configure the save path for video output:
output_directory = os.path.join(
    dir_path, "/weka/scratch/tshu2/sye10/habitat/outputs"
)  # @param {type:"string"}
output_path = os.path.join(dir_path, output_directory)
if not os.path.exists(output_path):
    os.mkdir(output_path)


if __name__ == "__main__":
    # Load the configuration file
    config_file_path = "/weka/scratch/tshu2/sye10/habitat/python_scripts/myEnvironment.yaml"
    config = get_config(config_file_path)

    # Initialize the Habitat environment
    env = habitat.Env(config=config)

    print("Environment initialized.")

    # Number of episodes to run
    num_episodes = 5
    agents = ["agent1", "agent2"]
 
    for episode in range(num_episodes):
        observations = env.reset()
        print(f"Episode {episode+1}/{num_episodes} started.")

        while not env.episode_over:
            # Process each agent in the environment
            for agent_id in agents:
                # Sample a random action for the current agent
                action = env.action_space.sample()
                # Step the environment with the action of the current agent
                observations = env.step(action, agent_id=agent_id)

                # Process observations for the current agent
                agent_position = observations[agent_id]["gps"]
                print(f"Agent {agent_id}'s current position: {agent_position}")

        print(f"Episode {episode+1} finished.\n")

    print("Simulation finished.")