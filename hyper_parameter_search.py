from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO
from cross_validation import *
import warnings
import torch
from sklearn.metrics import matthews_corrcoef
warnings.filterwarnings('ignore')
path_marker = '/'



def optimized(
    batch_size = 128,
    num_layers = 4,
    fingerprints_hidden_size = 64,
    lr = 10 **-4, 
    weight_decay = 10 **-2.5,
    dropout = 0.2,
    seed = 7):

  accuracy, sens, spec, auc, mcc = cross_validation(batch_size = round(batch_size), fingerprints_hidden_size = round(fingerprints_hidden_size), num_layers = round(num_layers), lr = round(lr, 2), weight_decay = round(weight_decay, 2), dropout = round(dropout, 2), seed = seed)
  return accuracy

if __name__ == "__main__":
  cov = matern32()
  gp = GaussianProcess(cov)
  acq = Acquisition(mode='UCB')
  param = {
          'dropout': ('cont', [0, 0.5]),
          'batch_size': ('int', [16, 256]),
          'num_layers': ('int', [1, 8]),
          'fingerprints_hidden_size': ('int', [64, 512]),
          'lr': ('cont', [2, 5]),
          'weight_decay': ('cont', [2, 5]),
          }
  gpgo = GPGO(gp, acq, optimized, param)
  gpgo.run(init_evals = 3, max_iter = 20)