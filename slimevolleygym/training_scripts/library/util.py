import argparse

# Example:
"""
from library import util

LOGDIR = util.get_logdir('model')
"""
def get_logdir(logdir):
  parser = argparse.ArgumentParser(description='train model')
  parser.add_argument('--logdir', help='path to save the best model while training', type=str, default=logdir)
  args = parser.parse_args()
  return args.logdir
