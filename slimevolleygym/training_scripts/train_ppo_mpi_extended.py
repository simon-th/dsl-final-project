#!/usr/bin/env python3
# trains slime agent from states with multiworker via MPI (fast wallclock time)
# run with
# mpirun -np 96 python train_ppo_mpi.py (replace 96 with number of CPU cores you have.)

import os
import gym
import slimevolleygym

from mpi4py import MPI
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import bench, logger, PPO1
from stable_baselines.common.callbacks import EvalCallback

NUM_TIMESTEPS = int(2e6)
SEED = 831
EVAL_FREQ = 200000
EVAL_EPISODES = 1000

# Log results
from library import util

args = util.get_args('ppo_extended', '../zoo/ppo/best_model.zip')
LOGDIR = args.logdir
if not os.path.exists(LOGDIR):
  os.makedirs(LOGDIR)

BEST_MODEL_PATH = args.modelpath
if not os.path.exists(BEST_MODEL_PATH):
  raise Exception('File does not exist:', BEST_MODEL_PATH)


def make_env(seed):
  env = gym.make("SlimeVolley-v0")
  env.seed(seed)
  return env

def train():
  """
  Train PPO1 model for slime volleyball, in MPI multiprocessing. Tested for 96 CPUs.
  """
  rank = MPI.COMM_WORLD.Get_rank()

  if rank == 0:
    logger.configure(folder=LOGDIR)

  else:
    logger.configure(format_strs=[])
  workerseed = SEED + 10000 * MPI.COMM_WORLD.Get_rank()
  set_global_seeds(workerseed)
  env = make_env(workerseed)

  env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
  env.seed(workerseed)

  model = PPO1.load(BEST_MODEL_PATH, env=env)

  eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

  model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

  env.close()
  del env
  if rank == 0:
    model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.


if __name__ == '__main__':
  train()
