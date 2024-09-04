from mt_dnn import *
from dataset import FingerPrintsDataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import warnings
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, recall_score, matthews_corrcoef, roc_auc_score
import pandas as pd
import os
warnings.filterwarnings('ignore')
path_marker = '/'

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        fp, Ys = batch
        outputs = model(fp)
        loss = loss_fn(torch.squeeze(outputs).to(torch.float32), Ys.to(torch.float32))
        loss.backward()
        optimizer.step()


def run_a_eval_epoch(model, validation_dataloader):
    true = []
    probe = []
    model.eval()
    with torch.no_grad():
      for i_batch, batch in enumerate(validation_dataloader):
        # model.zero_grad()
        fp, Ys = batch
        outputs = model(fp)
        true.append(Ys.data.numpy())
        probe.append(torch.squeeze(outputs).data.numpy())

    return true, probe


def cross_validation(
    batch_size = 43,
    num_workers = 4,
    epoch = 300,
    path = "E:/mt-dnn",
    path_split = "/",
    alpha_list = [0.787793427,	0.387688278,	0.517501716,	1.824489796],
    # alpha_list = [1.824489796],
    name = "init_ECFP.csv",
    fingerprints_size = 1024,
    fingerprints_hidden_size = 132,
    num_layers = 7,
    n_tasks = 4,
    patience = 40,
    lr = 3.17, 
    weight_decay = 4.50,
    dropout = 0.,
    seed = 7):
    if not os.path.exists(path + path_split + 'train_n_fold'):
      os.makedirs(path + path_split + 'train_n_fold')

    if not os.path.exists(path + path_split + 'valid_n_fold'):
      os.makedirs(path + path_split + 'valid_n_fold')

    if not os.path.exists(path + path_split + 'test_n_fold'):
      os.makedirs(path + path_split + 'test_n_fold')

    if os.path.exists(path + path_split + "cluster9"):
      n_fold = 10
    elif os.path.exists(path + path_split + "cluster4"):
      n_fold = 5
    elif os.path.exists(path + path_split + "cluster2"):
      n_fold = 3 
    else:
      raise FileNotFoundError("cluster csv was not found in %s"%path)

    stat_res = []
    set_random_seed(seed)

    for v in range(n_fold):
      folds_data = pd.DataFrame([])
      for i in range(n_fold):
        fold_data = pd.read_csv(path + path_split + "cluster" + str(i) + path_split + name, index_col = 0)\

        # modeling, perpare for "test_n_fold" file handly
        # folds_data = pd.concat([folds_data, fold_data], axis = 0)

        # pypermeter seaching
        if v == i :
          fold_data.to_csv(path + path_split + "test_n_fold" + path_split + name) 
        else:
            folds_data = pd.concat([folds_data, fold_data], axis = 0)

      folds_data.reset_index(drop = True, inplace = True)
      activity = folds_data["L858R/T790M/C797S"].tolist()
      x_t, x_r, y_t, y_r = train_test_split(folds_data.index, activity, train_size = 0.8, stratify= activity, random_state = 7)
      tr = folds_data.iloc[x_t].reset_index(drop = True)
      tr.to_csv(path + path_split + "train_n_fold" + path_split + name)
      va = folds_data.iloc[x_r].reset_index(drop = True)
      va.to_csv(path + path_split + "valid_n_fold" + path_split + name)   


      train_dataset = FingerPrintsDataset(path + path_split + "train_n_fold", name)
      valid_dataset = FingerPrintsDataset(path + path_split + "valid_n_fold", name)
      test_dataset = FingerPrintsDataset(path + path_split + "test_n_fold", name)



      train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, drop_last = True)
      valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
      valid_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)


      # model
      model = ANN(fingerprints_size = fingerprints_size,
                fingerprints_hidden_size = fingerprints_hidden_size,
                num_layers = num_layers,
                n_tasks = n_tasks,
                dropout = dropout)

      optimizer = torch.optim.Adam(model.parameters(), lr = 10**(-lr) , weight_decay = 10**(-weight_decay))
      filename = path + '/fold_save/{}_{}_{}_{}_{}_{}.pth'.format(v, batch_size, num_layers, fingerprints_hidden_size, lr, weight_decay)
      stopper = EarlyStopping(mode='higher', patience = patience, tolerance = 0, filename = filename)
      loss_fn = BalenceLoss(alpha_list = alpha_list)
      
      for epo in range(epoch):
        # train
        run_a_train_epoch(model, loss_fn, train_dataloader, optimizer)

        # validation
        train_true, train_probe = run_a_eval_epoch(model, train_dataloader)
        valid_true, valid_probe = run_a_eval_epoch(model, valid_dataloader)

        train_true =  np.concatenate(np.array(train_true), 0).flatten()
        train_probe =  np.concatenate(np.array(train_probe), 0).flatten()

        valid_true = np.concatenate(np.array(valid_true), 0).flatten()
        valid_probe = np.concatenate(np.array(valid_probe), 0).flatten()

        valid_probe = [valid_probe[i] for i in range(len(valid_probe)) if valid_true[i] == 0 or valid_true[i] == 1]
        valid_true = [valid_true[i] for i in range(len(valid_true)) if valid_true[i] == 0 or valid_true[i] == 1]
        
        train_probe = [train_probe[i] for i in range(len(train_probe)) if train_true[i] == 0 or train_true[i] == 1]
        train_true = [train_true[i] for i in range(len(train_true)) if train_true[i] == 0 or train_true[i] == 1]

        train_mcc = matthews_corrcoef(train_true, np.around(train_probe, 0))
        valid_mcc = matthews_corrcoef(valid_true, np.around(valid_probe, 0))
        early_stop = stopper.step(valid_mcc, model)

        # train_ba = balanced_accuracy_score(train_true, np.around(train_probe, 0))
        # valid_ba = balanced_accuracy_score(valid_true, np.around(valid_probe, 0))
        # early_stop = stopper.step(valid_ba, model)

        if early_stop:
            break

      # load the best model
      stopper.load_checkpoint(model)
      train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=collate_fn)
      valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=collate_fn)
      test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=collate_fn)


      train_true, train_probe = run_a_eval_epoch(model, train_dataloader)
      valid_true, valid_probe = run_a_eval_epoch(model, valid_dataloader)
      test_true, test_probe = run_a_eval_epoch(model, test_dataloader)

      # metrics
      train_true = np.around(np.concatenate(np.array(train_true), 0), 1).flatten()
      train_probe = np.concatenate(np.array(train_probe), 0).flatten()
      train_pred = np.around(train_probe, 0).flatten()

      valid_true = np.around(np.concatenate(np.array(valid_true), 0), 1).flatten()
      valid_probe = np.concatenate(np.array(valid_probe), 0).flatten()
      valid_pred = np.around(valid_probe, 0).flatten()

      test_true = np.around(np.concatenate(np.array(test_true), 0), 1).flatten()
      test_probe = np.concatenate(np.array(test_probe), 0).flatten()
      test_pred = np.around(test_probe, 0).flatten()

      test_pred = [test_pred[i] for i in range(len(test_pred)) if test_true[i] == 0 or test_true[i] == 1]
      test_probe = [test_probe[i] for i in range(len(test_probe)) if test_true[i] == 0 or test_true[i] == 1]
      test_true = [test_true[i] for i in range(len(test_true)) if test_true[i] == 0 or test_true[i] == 1]


      valid_pred = [valid_pred[i] for i in range(len(valid_pred)) if valid_true[i] == 0 or valid_true[i] == 1]
      valid_probe = [valid_probe[i] for i in range(len(valid_probe)) if valid_true[i] == 0 or valid_true[i] == 1]
      valid_true = [valid_true[i] for i in range(len(valid_true)) if valid_true[i] == 0 or valid_true[i] == 1]

      train_pred = [train_pred[i] for i in range(len(train_pred)) if train_true[i] == 0 or train_true[i] == 1]
      train_probe = [train_probe[i] for i in range(len(train_probe)) if train_true[i] == 0 or train_true[i] == 1]
      train_true = [train_true[i] for i in range(len(train_true)) if train_true[i] == 0 or train_true[i] == 1]

      pd_tr = pd.DataFrame({'train_true': train_true, 'train_probe': train_probe})
      pd_va = pd.DataFrame({'valid_true': valid_true, 'valid_probe': valid_probe})
      pd_te = pd.DataFrame({'test_true': test_true, 'test_probe': test_probe})

      pd_tr.to_csv('E:/mt-dnn/fold_stats/{}_{}_{}_{}_{}_{}_tr.csv'.format(v, batch_size, num_layers, fingerprints_hidden_size, lr, weight_decay), index=False)
      pd_va.to_csv('E:/mt-dnn/fold_stats/{}_{}_{}_{}_{}_{}_va.csv'.format(v, batch_size, num_layers, fingerprints_hidden_size, lr, weight_decay), index=False)
      pd_te.to_csv('E:/mt-dnn/fold_stats/{}_{}_{}_{}_{}_{}_te.csv'.format(v, batch_size, num_layers, fingerprints_hidden_size, lr, weight_decay), index=False)


      train_accuracy, train_sens, train_spec, train_auc, train_mcc = balanced_accuracy_score(train_true, train_pred), \
                                                      recall_score(train_true, train_pred), \
                                                      recall_score(train_true, train_pred, pos_label=0), \
                                                      roc_auc_score(train_true, train_probe), \
                                                      matthews_corrcoef(train_true, train_pred)
      valid_accuracy, valid_sens, valid_spec, valid_auc, valid_mcc = balanced_accuracy_score(valid_true,  valid_pred), \
                                                      recall_score(valid_true, valid_pred), \
                                                      recall_score(valid_true, valid_pred, pos_label=0), \
                                                      roc_auc_score(valid_true, valid_pred), \
                                                      matthews_corrcoef(valid_true, valid_pred)
      test_accuracy, test_sens, test_spec, test_auc, test_mcc = balanced_accuracy_score(test_true,  test_pred), \
                                                      recall_score(test_true, test_pred), \
                                                      recall_score(test_true, test_pred, pos_label=0), \
                                                      roc_auc_score(test_true, test_pred), \
                                                      matthews_corrcoef(test_true, test_pred)



      stat_res.append([v, 'train', train_accuracy, train_sens, train_spec, train_auc, train_mcc])
      stat_res.append([v, 'valid', valid_accuracy, valid_sens, valid_spec, valid_auc, valid_mcc])
      stat_res.append([v, 'test', test_accuracy, test_sens, test_spec, test_auc, test_mcc])

    stat_res_pd = pd.DataFrame(stat_res, columns=['repetition', 'group', 'accuracy', 'sens', 'spec', 'auc', 'mcc'])
    stat_res_pd.to_csv(path + '/fold_stats/{}_{}_{}_{}_{}_{}.csv'.format(v, batch_size, num_layers, fingerprints_hidden_size, lr, weight_decay), index=False)
    return stat_res_pd[stat_res_pd.group == 'test'][['accuracy', 'sens', 'spec', 'auc', 'mcc']].mean().values

if __name__ == "__main__":
  cross_validation()