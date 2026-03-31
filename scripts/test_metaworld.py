import gymnasium as gym
from gymnasium.wrappers import FrameStack, FlattenObservation
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN # type: ignore
from stable_baselines3.common.monitor import Monitor
import imageio

env_name = "peg-insert-side-v2"

if env_name + '-goal-observable' in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name+'-goal-observable'](render_mode='rgb_array')
    env._freeze_rand_vec = False
    env = FrameStack(env, 4)
    env = FlattenObservation(env)

env = Monitor(env)
obs = env.reset()
frames = []
for i in range(10):
    # Render the environment
    frame = env.render()
    frames.append(frame)
    # Take random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs = env.reset()

# Save frames as a gif
imageio.mimsave('metaworld_test.gif', frames, fps=5)

env.close()