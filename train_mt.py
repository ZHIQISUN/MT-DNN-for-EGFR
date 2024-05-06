from mt_dnn import *
from dataset import FingerPrintsDataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import time
import warnings
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score, matthews_corrcoef, roc_auc_score
import pandas as pd
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

if __name__ == '__main__':
    repetitions = 3
    batch_size = 43
    num_workers = 2
    n_tasks = 4
    alpha_list = [0.787793427,	0.387688278,	0.517501716,	1.824489796]
    name = "init_ECFP.csv"

    train_dataset = FingerPrintsDataset("E:/mt-dnn/train", name)
    valid_dataset = FingerPrintsDataset("E:/mt-dnn/valid", name)
    test_dataset = FingerPrintsDataset("E:/mt-dnn/test", name)

    stat_res = []
    for repetition_th in range(repetitions):
      set_random_seed(repetition_th)
      print('the number of train data:', len(train_dataset))
      print('the number of valid data:', len(valid_dataset))
      print('the number of test data:', len(test_dataset))
      train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                      collate_fn=collate_fn)
      valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=collate_fn)

      # model
      model = ANN(fingerprints_size = 1024,
                 fingerprints_hidden_size = 132,
                 num_layers = 7,
                 n_tasks = n_tasks,
                 dropout = 0.024)
      print('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
      if repetition_th == 0:
          print(model)
      optimizer = torch.optim.Adam(model.parameters(), lr=10 **-2.34 , weight_decay=10 **-4.22)
      dt = datetime.datetime.now()
      filename = 'E:/mt-dnn/model_save/{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond)
      stopper = EarlyStopping(mode='higher', patience=40, tolerance=0,
                              filename=filename)
      loss_fn = BalenceLoss(alpha_list = alpha_list)

      for epoch in range(500):
          st = time.time()
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

          end = time.time()
          if early_stop:
              break
          print(
              "epoch:%s \t train_mcc:%.4f \t valid_mcc:%.4f \t time:%.3f s" % (epoch, train_mcc, valid_mcc, end - st))

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

      pd_tr.to_csv('E:/mt-dnn/stats/{}_{:02d}_{:02d}_{:02d}_{:d}_tr.csv'.format(dt.date(), dt.hour, dt.minute, dt.second,dt.microsecond), index=False)
      pd_va.to_csv('E:/mt-dnn/stats/{}_{:02d}_{:02d}_{:02d}_{:d}_va.csv'.format(dt.date(), dt.hour, dt.minute, dt.second,dt.microsecond), index=False)
      pd_te.to_csv('E:/mt-dnn/stats/{}_{:02d}_{:02d}_{:02d}_{:d}_te.csv'.format(dt.date(), dt.hour, dt.minute, dt.second,dt.microsecond), index=False)

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
      test_accuracy, test_sens, test_spec, test_auc, test_mcc     = balanced_accuracy_score(test_true, test_pred), \
                                                      recall_score(test_true, test_pred), \
                                                      recall_score(test_true, test_pred, pos_label=0), \
                                                      roc_auc_score(test_true, test_probe), \
                                                      matthews_corrcoef(test_true, test_pred)

      print('***best %s model***' % repetition_th)
      print("train_accuracy:%.4f \t train_sens:%.4f \t train_spec:%.4f \t train_auc:%.4f \t train_mcc:%.4f" % (
          train_accuracy, train_sens, train_spec, train_auc, train_mcc))
      print("valid_accuracy:%.4f \t valid_sens:%.4f \t valid_spec:%.4f \t valid_auc:%.4f \t valid_mcc:%.4f" % (
          valid_accuracy, valid_sens, valid_spec, valid_auc, valid_mcc))
      print("test_accuracy:%.4f \t test_sens:%.4f \t test_spec:%.4f \t test_auc:%.4f \t test_mcc:%.4f" % (
          test_accuracy, test_sens, test_spec, test_auc, test_mcc))
      stat_res.append([repetition_th, 'train', train_accuracy, train_sens, train_spec, train_auc, train_mcc])
      stat_res.append([repetition_th, 'valid', valid_accuracy, valid_sens, valid_spec, valid_auc, valid_mcc])
      stat_res.append([repetition_th, 'test', test_accuracy, test_sens, test_spec, test_auc, test_mcc])

    stat_res_pd = pd.DataFrame(stat_res, columns=['repetition', 'group', 'accuracy', 'sens', 'spec', 'auc', 'mcc'])
    stat_res_pd.to_csv(
        'E:/mt-dnn/stats/{}_{:02d}_{:02d}_{:02d}_{:d}.csv'.format(dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond),
        index=False)
    print(stat_res_pd[stat_res_pd.group == 'train'][['accuracy', 'sens', 'spec', 'auc', 'mcc']].mean().values,
          stat_res_pd[stat_res_pd.group == 'train'][['accuracy', 'sens', 'spec', 'auc', 'mcc']].std().values)
    print(stat_res_pd[stat_res_pd.group == 'valid'][['accuracy', 'sens', 'spec', 'auc', 'mcc']].mean().values,
          stat_res_pd[stat_res_pd.group == 'valid'][['accuracy', 'sens', 'spec', 'auc', 'mcc']].std().values)
    print(stat_res_pd[stat_res_pd.group == 'test'][['accuracy', 'sens', 'spec', 'auc', 'mcc']].mean().values,
          stat_res_pd[stat_res_pd.group == 'test'][['accuracy', 'sens', 'spec', 'auc', 'mcc']].std().values)