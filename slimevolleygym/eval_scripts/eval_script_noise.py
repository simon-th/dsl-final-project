"""
Multiagent example.

Evaluate the performance of different trained models in zoo against each other.

This file can be modified to test your custom models later on against existing models.

Model Choices
=============

BaselinePolicy: Default built-in opponent policy (trained in earlier 2015 project)

baseline: Baseline Policy (built-in AI). Simple 120-param RNN.
ppo: PPO trained using 96-cores for a long time vs baseline AI (train_ppo_mpi.py)
cma: CMA-ES with small network trained vs baseline AI using estool
ga: Genetic algorithm with tiny network trained using simple tournament selection and self play (input x(train_ga_selfplay.py)
random: random action agent
"""

import gym
import os
import numpy as np
import argparse
import pathlib
import slimevolleygym
from slimevolleygym.mlp import makeSlimePolicy, makeSlimePolicyLite # simple pretrained models
from slimevolleygym import BaselinePolicy
from time import sleep
from collections import defaultdict
import pandas as pd
from training_scripts.library.action_wrapper import ConstantNoiseActionWrapperMulti
from training_scripts.library.Observation_Wrapper import ObservationWrapperMulti

#import cv2

import warnings
# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
# Filter tensorflow version warnings
# https://github.com/hill-a/stable-baselines/issues/298#issuecomment-637613817
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

from stable_baselines import PPO1

np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)

class PPOPolicy:
  def __init__(self, path):
    print(path)
    self.model = PPO1.load(path)

  def predict(self, obs):
    action, state = self.model.predict(obs, deterministic=True)
    return action

class RandomPolicy:
  def __init__(self):
    self.action_space = gym.spaces.MultiBinary(3)
    pass

  def predict(self, obs):
    return self.action_space.sample()

def makeBaselinePolicy(_):
  return BaselinePolicy()

def rollout(env, policy0, policy1, render_mode=False):
  """ play one agent vs the other in modified gym-style loop. """
  obs0 = env.reset()
  obs1 = obs0 # same observation at the very beginning for the other agent

  done = False
  total_reward = 0
  #count = 0

  while not done:

    # print(obs0)
    action0 = policy0.predict(obs0)
    action1 = policy1.predict(obs1)

    # uses a 2nd (optional) parameter for step to put in the other action
    # and returns the other observation in the 4th optional "info" param in gym's step()
    obs0, reward, done, info = env.step(action0, action1)
    obs1 = info['otherObs']

    # print("right:", obs0)
    # print("left:", obs1)
    total_reward += reward

    if render_mode:
      env.render()
      """ # used to render stuff to a gif later.
      img = env.render("rgb_array")
      filename = os.path.join("gif","daytime",str(count).zfill(8)+".png")
      cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
      count += 1
      """
      sleep(0.01)

    # sleep(0.01)

  return total_reward

def evaluate_multiagent(env, policy0, policy1, render_mode=False, n_trials=1000, init_seed=721):
  history = []
  for i in range(n_trials):
    env.seed(seed=init_seed+i)
    cumulative_score = rollout(env, policy0, policy1, render_mode=render_mode)
    history.append(cumulative_score)
  return history

def evaluate_agents(test_agents, benchmark_agents, render_mode=False, n_trials=1000, init_seed=721):
  results = defaultdict(list)
  for test_model_name, test_agent in test_agents.items():
    for benchmark_model_name, benchmark_agent in benchmark_agents.items():
      env = ObservationWrapperMulti(ConstantNoiseActionWrapperMulti(gym.make("SlimeVolley-v0")))
      env.seed(args.seed)


      history = evaluate_multiagent(env, test_agent, benchmark_agent,
        render_mode=render_mode, n_trials=args.trials, init_seed=args.seed)

      # print("history dump:", history)

      print(test_model_name, "vs", benchmark_model_name, ":", np.round(np.mean(history), 3), "±", np.round(np.std(history), 3))

      results[benchmark_model_name + "_mean"].append(np.round(np.mean(history), 3))
      results[benchmark_model_name + "_std"].append(np.round(np.std(history), 3))

  results_df = pd.DataFrame(results, index=test_agents.keys())
  str_results_df = pd.DataFrame()
  for offset in range(0, results_df.shape[1], 2):
      str_results_df[results_df.columns[offset].rsplit('_', 1)[0]] = results_df.apply(make_func(offset), axis=1)

  return results_df, str_results_df

def make_func(offset=0):
    def func(x):
        return '{} ± {}'.format(x[0 + offset], x[1 + offset])
    return func

def print_and_save_results(results_df, str_results_df, resultfilename):
  results_df.to_csv(results_dir / "{}.csv".format(resultfilename))
  print(results_df)
  str_results_df.to_csv(results_dir / "{}_str.csv".format(resultfilename))
  print(str_results_df)


if __name__=="__main__":

  APPROVED_MODELS = ["baseline", "ppo", "ga", "cma", "random"]

  def checkchoice(choice):
    choice = choice.lower()
    if choice not in APPROVED_MODELS:
      return False
    return True

  parent_dir = pathlib.Path(__file__).parent
  results_dir = parent_dir / "results"
  zoo_dir = parent_dir / "zoo"

  parser = argparse.ArgumentParser(description='Evaluate pre-trained agents against each other.')
  parser.add_argument('--zoodir', help='path of zoo dir', type=str, default=zoo_dir)
  parser.add_argument('--resultdir', help='path of result dir', type=str, default=results_dir)
  parser.add_argument('--resultfilename', help='name of result csv', type=str, default="results")
  parser.add_argument('--benchmark', help='set of agents to benchmark against (pretrained or extended)', type=str, default="")
  parser.add_argument('--evaltest', action='store_true', help='evaluate each test agent against itself and other test agents (may not be used with --benchmark)', default=False)
  parser.add_argument('--render', action='store_true', help='render to screen (not recommended with high number of trials)', default=False)
  parser.add_argument('--day', action='store_true', help='daytime colors?', default=False)
  parser.add_argument('--pixel', action='store_true', help='pixel rendering effect? (note: not pixel obs mode)', default=False)
  parser.add_argument('--seed', help='random seed (integer)', type=int, default=721)
  parser.add_argument('--trials', help='number of trials (default 500)', type=int, default=500)

  args = parser.parse_args()

  if args.benchmark and args.evaltest:
    raise Exception('only one of --benchmark or --evaltest may be used')
  elif not args.benchmark and not args.evaltest:
    raise Exception('must set either --benchmark or --evaltest')

  if args.day:
    slimevolleygym.setDayColors()

  if args.pixel:
    slimevolleygym.setPixelObsMode()

  render_mode = args.render

  results_dir = args.resultdir
  if not os.path.exists(results_dir):
    os.makedirs(results_dir)

  zoo_dir = args.zoodir
  if not os.path.exists(zoo_dir):
    raise Exception('zoo directory does not exist')

  # zoo directory must contain pretrained and extended models
  pretrained_dir = zoo_dir / "pretrained"
  extended_dir = zoo_dir / "extended"
  # Example:
  # reward_wrapper_dir = zoo_dir / "reward_wrapper"
  obs_small_always_dir = zoo_dir / "obs_small_always"

  # Add the test agents you would like to evaluate to the test_agents dictionary { "model_name": "path/to/model"}
  # You may delete the existing ones in the test_agents dictionary (they are just examples)
  test_agents = {
      "ppo": PPOPolicy(obs_small_always_dir / "ppo.zip"),
      "ppo_sp": PPOPolicy(obs_small_always_dir / "ppo_sp.zip"),
      "ga_sp": makeSlimePolicyLite(obs_small_always_dir / "ga_sp.json"),
  }

  if args.benchmark and not args.evaltest:
    pretrained_agents = {
      "baseline": BaselinePolicy(),
      "ppo": PPOPolicy(pretrained_dir / "ppo.zip"),
      "ppo_sp": PPOPolicy(pretrained_dir / "ppo_sp.zip"),
      "ga_sp": makeSlimePolicyLite(pretrained_dir / "ga_sp.json"),
      "random": RandomPolicy(),
    }

    extended_agents = {
      "ppo_extended": PPOPolicy(extended_dir / "ppo.zip"),
      "ppo_sp_extended": PPOPolicy(extended_dir / "ppo_sp.zip"),
      "ga_sp_extended": makeSlimePolicyLite(extended_dir / "ga_sp.json"),
    }

    benchmarks = {
      "pretrained": pretrained_agents,
      "extended": extended_agents
    }

    benchmark_agents = benchmarks[args.benchmark]

    results_df, str_results_df = evaluate_agents(test_agents, benchmark_agents, render_mode, args.trials, args.seed)
    print_and_save_results(results_df, str_results_df, args.resultfilename)

  elif args.evaltest and not args.benchmark:
    results_df, str_results_df = evaluate_agents(test_agents, test_agents, render_mode, args.trials, args.seed)
    print_and_save_results(results_df, str_results_df, args.resultfilename)

