### CLASSIFIER V1: recognization of in-painting technique

## HOW TO DEMO LOCALE
* Creation of file SPAN

```
python3 -u compute_SPAN_pesi_originali.py  >  compute_SPAN_pesi_originali.txt &
```
* Creation models 

```
python3 -u compute_MODELS_pesi_originali.py  >  compute_MODELS_pesi_originali.txt &
```
* Run classifier_v1

```
python3 -u classifier_v1_pesi_originali.py    >  classifier_v1_pesi_originali.txt &
```
* Read results from class_mat_results_pesi_originali.mat. We obtain the accuracy by counting the number of occorrences of each technique and dividing it for 31 (numbers of test video)
```
python3 read.py
```
* Compute the confusion matrix by putting the previous results in the array of file confusion_matrix.py
 ```
python3 confusion_matrix.py
```
