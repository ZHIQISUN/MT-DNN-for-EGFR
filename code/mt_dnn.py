import torch
import torch.nn as nn
from dgl.nn.pytorch import edge_softmax
import random
import numpy as np
import datetime
import torch.nn.functional as F

class ANN(nn.Module):

    def __init__(self,
                 fingerprints_size = 1024,
                 fingerprints_hidden_size = 132,
                 num_layers = 7,
                 n_tasks = 4,
                 dropout = 0.):
        super(ANN, self).__init__()
        self.input_fp_size = fingerprints_size
        self.fingerprints_size = fingerprints_size
        self.fingerprints_hidden_size = fingerprints_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        fingerprints_hidden = []

        fingerprints_hidden.append(
            nn.Sequential(
              nn.Dropout(dropout),
              nn.Linear(self.fingerprints_size, self.fingerprints_hidden_size),
              nn.BatchNorm1d(self.fingerprints_hidden_size),
              nn.ReLU()
              )
          )
        for lay in range(self.num_layers):
          fingerprints_hidden.append(
            nn.Sequential(
              nn.Dropout(dropout),
              nn.Linear(self.fingerprints_hidden_size, self.fingerprints_hidden_size),
              nn.BatchNorm1d(self.fingerprints_hidden_size),
              nn.ReLU()
              )
          )

        self.fingerprints_hidden = nn.ModuleList(fingerprints_hidden)

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.fingerprints_hidden_size, n_tasks),
            nn.Sigmoid()
        )

    def forward(self, feat):
        fingerprints = feat
        for lay in self.fingerprints_hidden:
          fingerprints = lay(fingerprints)
        return self.predict(fingerprints)

class BalenceLoss(nn.Module):
    def __init__(self, alpha_list = [1, 1, 1, 1, 1, 1,], logits = True, reduce = True):
        super(BalenceLoss, self).__init__()
        self.alpha_list = alpha_list
        self.logits = logits
        self.reduce = reduce

    def forward(self, pred, targets):
        balence_loss = torch.zeros(pred.shape[0])
        for i in range(len(self.alpha_list)):
          pr = pred[:,i]
          ta = targets[:,i]
          # pr = pred
          # ta = targets
          if self.logits:
              BCE_loss = F.binary_cross_entropy_with_logits(pr, ta, reduce=False)
          else:
              BCE_loss = F.binary_cross_entropy(pr, ta, reduce=False)
          balence_weight = torch.tensor([1, self.alpha_list[i], 0])
          alpha = balence_weight[ta.int()]
          balence_loss += alpha * BCE_loss
        if self.reduce:
            return torch.mean(balence_loss)
        else:
            return balence_loss



def collate_fn(data_batch):
    fingerprints, Ys = map(list, zip(*data_batch))
    fp = torch.tensor(fingerprints,dtype=torch.float)
    Ys = torch.stack(Ys, dim = 0)
    return fp, Ys

def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping(object):
    def __init__(self,  mode='higher', patience=20, filename=None, tolerance=0.0):
        if filename is None:
            dt = datetime.datetime.now()
            filename = 'E:/model_data/model_save/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)

            self.counter = 0
        else:
            self.counter += 1
            # print(
            #     f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)  

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])
 