import argparse

# Example:
"""
from library import util

args = util.get_args('model_name', 'path/to/model')

LOGDIR = args.logdir
if not os.path.exists(LOGDIR):
  os.makedirs(LOGDIR)

MODEL_PATH = args.modelpath   # may want to check if file exists before loading model
"""
def get_args(logdir, modelpath=""):
  parser = argparse.ArgumentParser(description='train model')
  parser.add_argument('--logdir', help='path to save the best model while training', type=str, default=logdir)
  parser.add_argument('--modelpath', help='path to load model from (used for loading a pretrained model)', type=str, default=modelpath)
  args = parser.parse_args()

  return args
