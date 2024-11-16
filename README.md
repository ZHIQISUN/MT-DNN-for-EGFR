## MT-DNN-for-EGFR
### introduction
The emergence of T790M and C797S resistance mutations has led to sequential failure of existing clinical epidermal growth factor receptor tyrosine kinase (EGFR) inhibitors. Currently, the fourth-generation EGFR inhibitors targeting L858R/T790M/C797S mutations are in clinical trials mostly and it is necessary to study and develop new fourth-generation EGFR inhibitors. This study establishes a structure-activity relationship model for multi-generation EGFR inhibitors based on a multi-task deep neural network (MT-DNN). An internal dataset containing 2302 multi-target EGFR inhibitors targeting wild type (83%) ，L858R (92%), L858R/T790M (96%),and L858R/T790M/C797S (60%) mutations was collected. The MT-DNN model achieves an average AUC (area under the Receiver Operating Characteristic curve) of 0.88 in cluster cross-validation of the internal dataset. In order to compare the prediction performance on the external dataset of 304 fourth-generation EGFR inhibitors, we also constructed single task models of different algorithms based on 1384 L858R/T790M/C797S (60%) mutation inhibitors. MT-DNN’s performance far exceeds them after adding multi-generation EGFR inhibitors related tasks. The highly relevant information learned from multi-generation EGFR inhibitors makes MT-DNN significantly different from single target deep neural network (ST-DNN). Furthermore, superior to other interpretability models that only provide case studies, the combined application of MT-DNN and interpretability analysis offers a rigorous structural information from a global perspective for development of multi-generation EGFR inhibitors. Overall, MT-DNN model can mine core scaffold and important fragments of multi-generation EGFR inhibitors by comparing similarities and differences of multi-tasks, providing valuable information from a structure-activity relationship perspective to address resistant mutation problem.
### notes
The file location information in the code has not been changed. If you need to use the code, please change the location of all files yourself.The environment of this code is windows, if you need to use the code, please change the location of all files by yourself. Due to the small size of the project, the use of GPU is not considered.
### Guide
The best model results are stored in `fold_save/best_model/*pth`. Anyone can download the pth model file and use it to predict new EGFR inhibitors. The input format of the file refers to the csv file in `data/test_ECFP.csv`, and the code used to predict can refer to `code/result.ipynb` file.


