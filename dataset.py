import torch
import pandas as pd
from train_valid_test_split import *
import warnings
warnings.filterwarnings('ignore')


class FingerPrintsDataset(object):
  def __init__(self, path, name, path_split = "/"):
    self.data = pd.read_csv(path + path_split + name, index_col=[0])
    self.fingerprints = [self.data.iloc[i,:1024].tolist() for i in range(self.data.shape[0])]
    self.activities = [self.data.iloc[i,1024:-1].tolist() for i in range(self.data.shape[0])]
    # self.activities = [self.data.iloc[i,-2].tolist() for i in range(self.data.shape[0])]


  
  def __getitem__(self, i):
      return self.fingerprints[i], torch.tensor(self.activities[i])
  def __len__(self):
      return len(self.activities)


if __name__ == "__main__":
  splits = Dataset_Split("E:/mt-dnn", "init_ECFP.csv", split_method='stratify_split')
  splits.split_to_disk()  
