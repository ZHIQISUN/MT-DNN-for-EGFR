import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

class Dataset_Split(object):
  def __init__(self, path, name, valid_size = 0.2, test_size = 0.2, split_method = 'random_split', random_state = 0, path_split = "/"):
    self.path_split = path_split
    self.random_state = random_state
    self.path = path
    self.name = name
    self.data = pd.read_csv(self.path + self.path_split + "csv" + self.path_split + self.name, index_col=[0])
    self.activities = np.array([self.data.iloc[i,1024:-1].tolist() for i in range(self.data.shape[0])])
    self.classify = np.array([self.data.iloc[i,-1].tolist() for i in range(self.data.shape[0])])
    self.index = [i for i in range(self.data.shape[0])]


    self.valid_size = valid_size
    self.test_size = test_size
    assert split_method in ['random_split', 'stratify_split', 'scaffold_split']
    self.split_method = split_method


  def random_split(self):
    X_train, X_res, y_train, y_res = train_test_split(self.index, self.activities, train_size = 1-self.valid_size-self.test_size, random_state = self.random_state)
    X_valid, X_test, y_valid, y_test = train_test_split(X_res, y_res, train_size = self.valid_size/(self.test_size+self.valid_size),  random_state = self.random_state)

    return X_train, X_valid, X_test
  
  def stratify_split(self):
    
    # X_train, X_res, y_train, y_res = train_test_split(self.index, self.activities, train_size = 1-self.valid_size-self.test_size, stratify=[1 if sum(self.activities[i]) > 1 else 0 for i in range(len(self.activities))], random_state = self.random_state)
    # X_valid, X_test, y_valid, y_test = train_test_split(X_res, y_res, train_size = self.valid_size/(self.test_size+self.valid_size), stratify=[1 if sum(y_res[i]) > 1 else 0 for i in range(len(y_res))], random_state = self.random_state)
    X_train, X_res, y_train, y_res = train_test_split(self.index, self.activities, train_size = 1-self.valid_size-self.test_size, stratify=self.classify, random_state = self.random_state)
    res_classify = np.array(self.data.iloc[X_res,-1])
    X_valid, X_test, y_valid, y_test = train_test_split(X_res, y_res, train_size = self.valid_size/(self.test_size+self.valid_size), stratify=res_classify, random_state = self.random_state)
    
    return X_train, X_valid, X_test
 
 
  
  def split_to_disk(self):
    if not os.path.exists(self.path + self.path_split + 'train'):
      os.makedirs(self.path + self.path_split + 'train')

    if not os.path.exists(self.path + self.path_split + 'valid'):
      os.makedirs(self.path + self.path_split + 'valid')

    if not os.path.exists(self.path + self.path_split + 'test'):
      os.makedirs(self.path + self.path_split + 'test')

    if self.split_method == 'random_split':
      X_train, X_valid, X_test = self.random_split()
      
    elif self.split_method == 'stratify_split':
      X_train, X_valid, X_test = self.stratify_split()
    
    print("train_set:%s  valid_set:%s  test_set:%s"%(len(X_train), len(X_valid), len(X_test)))
    
    # static_train_x = [self.data.iloc[i,:-3] for i in X_train]
    # static_x = [self.data.iloc[i,:-3] for i in range(self.data.shape[0])]
    # scaler = preprocessing.MinMaxScaler().fit(static_train_x)
    # static_x = scaler.transform(static_x) *0.8 + 0.1
    # for i in range(len(static_x)):
    #   self.data.iloc[i,:-3] = static_x[i]

    train_data = pd.DataFrame(self.data.iloc[X_train])
    train_data.to_csv(self.path + self.path_split + 'train' + self.path_split + self.name)
    valid_data = pd.DataFrame(self.data.iloc[X_valid])
    valid_data.to_csv(self.path + self.path_split + 'valid' + self.path_split + self.name)
    test_data = pd.DataFrame(self.data.iloc[X_test])
    test_data.to_csv(self.path + self.path_split + 'test' + self.path_split + self.name)