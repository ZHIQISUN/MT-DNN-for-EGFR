import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

def tanimoto_distance_matrix(fp_list):
    dissimilarity_matrix = []
    for i in range(1, len(fp_list)):
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix

def cluster_fingerprints(fingerprints, cutoff=0.2):
    distance_matrix = tanimoto_distance_matrix(fingerprints)
    clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
    return clusters
    

class Crossingvalid_Split(object):
  def __init__(self, path, name, name2, fold = 5, random_state = 666, path_split = "/"):
    self.path_split = path_split
    self.random_state = random_state
    self.path = path
    self.name = name
    self.name2 = name2
    self.data = pd.read_csv(self.path + self.path_split + "csv" + self.path_split + self.name, index_col=[0])
    self.data2 = pd.read_csv(self.path + self.path_split + "csv" + self.path_split + self.name2, index_col=[0])
    self.fold = fold
    self.fingerprints = [self.data.iloc[i,:1024].tolist() for i in range(self.data.shape[0])]
  
  def butina_split(self, cutoff=0.2):
    self.data2['ROMol'] = self.data2['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
    fingerprints = self.data2["ROMol"].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)).tolist()
    clusters = cluster_fingerprints(fingerprints, cutoff=cutoff)
    clusters = np.array(clusters, dtype=object)
    np.random.shuffle(clusters)
    splits = [[] for i in range(self.fold)]
    n = 0
    for cluster in clusters:
      if n < self.fold:
        splits[n].extend(cluster)
      else:
        splits[self.fold - 1].extend(cluster)
      if len(splits[n]) >= len(self.fingerprints)/self.fold:
        n += 1
    return splits
  
  def split_to_disk(self, cutoff = 0.2):
    splits = self.butina_split(cutoff = cutoff)
    for i in range(len(splits)):
      if not os.path.exists(self.path + self.path_split + 'cluster' + str(i)):
        os.makedirs(self.path + self.path_split + 'cluster' + str(i))
      cluster =  pd.DataFrame(self.data.iloc[splits[i]])
      cluster.to_csv(self.path + self.path_split + 'cluster' + str(i) + self.path_split + self.name)
      print("cluster%s: %s"%(i,len(splits[i])), end = " ")

if __name__ == "__main__":
  splits = Crossingvalid_Split("E:/mt-dnn", "init_ECFP.csv", "data2.csv", fold = 5, random_state = 7)
  splits.split_to_disk(cutoff = 0.2)  