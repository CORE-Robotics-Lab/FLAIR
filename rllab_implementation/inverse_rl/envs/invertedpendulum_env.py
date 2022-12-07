import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env


import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class CustomInvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, max_timesteps=100):
        self.timesteps = 0
        self.max_timesteps = max_timesteps
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = -np.abs(ob[1])
        self.timesteps += 1
        done = self.timesteps > self.max_timesteps
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def reset(self):
        self.timesteps = 0
        return super(CustomInvertedPendulumEnv, self).reset()

    def log_diagnostics(self, paths):
        pass
        # forward_rew = np.array([np.mean(traj['env_infos']['reward_forward']) for traj in paths])
        # reward_ctrl = np.array([np.mean(traj['env_infos']['reward_ctrl']) for traj in paths])
        # reward_cont = np.array([np.mean(traj['env_infos']['reward_contact']) for traj in paths])
        # reward_flip = np.array([np.mean(traj['env_infos']['reward_flipped']) for traj in paths])
        #
        # logger.record_tabular('AvgRewardFwd', np.mean(forward_rew))
        # logger.record_tabular('AvgRewardCtrl', np.mean(reward_ctrl))
        # logger.record_tabular('AvgRewardContact', np.mean(reward_cont))
        # logger.record_tabular('AvgRewardFlipped', np.mean(reward_flip))


if __name__ == "__main__":
    env = CustomInvertedPendulumEnv()
    env.reset()
    for i in range(10000):
        env.render()
        ob, rew, done, info = env.step(env.action_space.sample())
        print(i, rew)
        if done:
            break
