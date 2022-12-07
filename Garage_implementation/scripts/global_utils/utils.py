import pickle
import os

import numpy as np

def load_expert_from_core_MSD(filename, length, repeat_each_skill, separate_styles, squeeze_option=2):
    experts_multi_styles = []
    with open(filename, "rb") as f:
        trajectories = pickle.load(f)
        for trajectory in trajectories:
            experts = []
            for i in range(repeat_each_skill):
                actions = np.array(trajectory["actions"])
                if squeeze_option == 0:
                    actions = actions.squeeze(axis=2)
                elif squeeze_option == 1:
                    actions = np.expand_dims(actions, axis=1)
                elif squeeze_option == 2:
                    actions = actions.squeeze(axis=1)
                experts.append({"observations": np.array(trajectory["observations"][i*length:(i+1)*length]),
                                "actions": actions[i*length:(i+1)*length]})
            if separate_styles:
                experts_multi_styles.append(experts)
            else:
                experts_multi_styles.extend(experts)
    return experts_multi_styles

def load_pickle_trajectories(filename, repeat_each_skill, separate_styles):
    experts_multi_styles = []
    with open(filename, "rb") as f:
        trajectories = pickle.load(f)['paths']
        print(np.array(trajectories).shape)
        for trajectory in trajectories:
            experts = []
            for i in range(repeat_each_skill):
                experts.append({"observations": np.array(trajectory["observations"][i*1000:(i+1)*1000]),
                                "actions": np.array(trajectory["actions"][i*1000:(i+1)*1000])})
            if separate_styles:
                experts_multi_styles.append(experts)
            else:
                experts_multi_styles.extend(experts)
    return experts_multi_styles
"""
def load_expert_from_core_MSD(filename, repeat_each_skill, separate_styles):
    experts_multi_styles = []
    with open(filename, "rb") as f:
        trajectories = pickle.load(f)
        for trajectory in trajectories:
            experts = []
            for i in range(repeat_each_skill):
                experts.append({"observations": trajectory["observations"][i*1000:(i+1)*1000],
                                "actions": trajectory["actions"][i*1000:(i+1)*1000]})
            if separate_styles:
                experts_multi_styles.append(experts)
            else:
                experts_multi_styles.extend(experts)
    return experts_multi_styles
"""

def load_expert_from_core_MSD_only_two_skills(filename, repeat_each_skill, separate_styles):
    experts_multi_styles = []
    with open(filename, "rb") as f:
        trajectories = pickle.load(f)
        for idx, trajectory in enumerate(trajectories):
            if idx == 4 or idx == 13:
                experts = []
                for i in range(repeat_each_skill):
                    experts.append({"observations": trajectory["observations"][i*1000:(i+1)*1000],
                                    "actions": trajectory["actions"][i*1000:(i+1)*1000]})
                if separate_styles:
                    experts_multi_styles.append(experts)
                else:
                    experts_multi_styles.extend(experts)
    return experts_multi_styles


def load_expert_from_core_MSD_only_two_skills_without_skill_label(filename, repeat_each_skill):
    experts_multi_styles = []
    with open(filename, "rb") as f:
        trajectories = pickle.load(f)
        for idx, trajectory in enumerate(trajectories):
            if idx == 4 or idx == 13:
                for i in range(repeat_each_skill):
                    experts = {"observations": trajectory["observations"][i*1000:(i+1)*1000],
                               "actions": trajectory["actions"][i*1000:(i+1)*1000]}
                    experts_multi_styles.append(experts)
    return experts_multi_styles


def get_test_data(dataset_filename, num_skills):
    """
    returned data is a map of states, actions, and ground_truth_rewards
    the shape of each one is [num_skills * num_agents (typically 1) * num_noise_levels, length_of_trajectory, dim_{state,action, reward}]

    :param dataset_filename:
    :param num_skills:
    :return:
    """
    with open(dataset_filename, "rb") as f:
        train_trajectories = pickle.load(f)
    data = {"states": [],
            "actions": [],
            "ground_truth_rewards": []}
    for skill in range(num_skills):
        for agent_trajectory in train_trajectories[skill]:
            for states, actions, rewards in agent_trajectory:
                data["states"].append(states)
                data["actions"].append(actions)
                data["ground_truth_rewards"].append(np.sum(rewards))
    for key in data.keys():
        data[key] = np.array(data[key])
    print("data obs shape: ", data["states"].shape)
    return data


def slice_test_data(raw_data, slice):
    new_data = {}
    for key in raw_data.keys():
        new_data[key] = raw_data[key][slice]
    return new_data


def save_video(ims, filename, fps=30.0):
    import cv2
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()
