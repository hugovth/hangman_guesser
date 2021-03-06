# import base64
# import imageio
from hangman_guesser.env.TFenv import HangmanEnvironment

import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.networks.q_network import QNetwork

from tf_agents.environments import suite_gym


from tf_agents.environments import tf_py_environment

from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from multiprocessing import Pool

# Globals
NUMBER_EPOSODES = 20000
COLLECTION_STEPS = 1
BATCH_SIZE = 64
EVAL_EPISODES = 10
EVAL_INTERVAL = 1000


train_env = HangmanEnvironment()
evaluation_env = HangmanEnvironment()

print("Observation Spec:")
print(train_env.time_step_spec().observation)

print("Reward Spec:")
print(train_env.time_step_spec().reward)

print("Action Spec:")
print(train_env.action_spec())

train_env = tf_py_environment.TFPyEnvironment(train_env)
evaluation_env = tf_py_environment.TFPyEnvironment(evaluation_env)


hidden_layers = (100,)

# dqn_network = QNetwork(
#     train_env.observation_spec(),
#     train_env.action_spec(),
#     fc_layer_params=hidden_layers)

ddqn_network = QNetwork(
    train_env.observation_spec(), train_env.action_spec(), fc_layer_params=hidden_layers
)

counter = tf.Variable(0)

ddqn_agent = DdqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=ddqn_network,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=counter,
)

ddqn_agent.initialize()


def get_average_reward(environment, policy, episodes=10):

    total_reward = 0.0

    for _ in range(episodes):
        time_step = environment.reset()
        episode_reward = 0.0

    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_reward += time_step.reward

    total_reward += episode_reward
    avg_reward = total_reward / episodes

    return avg_reward.numpy()[0]


class ExperienceReplay(object):
    def __init__(self, agent, enviroment):
        self._replay_buffer = TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=enviroment.batch_size,
            max_length=50000,
        )

        self._random_policy = RandomTFPolicy(
            train_env.time_step_spec(), enviroment.action_spec()
        )

        self._fill_buffer(train_env, self._random_policy, steps=100)

        self.dataset = self._replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=BATCH_SIZE, num_steps=2
        ).prefetch(3)

        self.iterator = iter(self.dataset)

    def _fill_buffer(self, enviroment, policy, steps):
        for _ in range(steps):
            self.timestamp_data(enviroment, policy)

    def timestamp_data(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        timestamp_trajectory = trajectory.from_transition(
            time_step, action_step, next_time_step
        )

        self._replay_buffer.add_batch(timestamp_trajectory)
