# NN-log-reg

description
-----------
This neural network uses normalized gene expression data (e.g. of scRNAseq data) to predict whether a cell or a sample 
belongs to a class and which transcripts are important for the prediction of the neural network.
Alternatively, the data can be anlyzed by SVC, XGBoost boosted tree-based classification [1], logistic regression, a standard neural network or TabNet [2].
<br />
The method was developed for scRNAseq of Cutaneous T Cell Lymphoma cells, Chang YT et al., 'MHC-I Upregulation Safeguards Neoplastic T Cell in the Skin Against NK Cell Eradication in Mycosis Fungoides', in submission. 

[1] Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. ArXiv.

[2] Arik, S. Ö., & Pfister, T. (2019). TabNet : Attentive Interpretable Tabular Learning. ArXiv.

input data
-----------
data_frame:    data_frame: columns are cells/samples, rows are genes 
<br />
label_file:    data frame of labels (column headings are cells/samples)


system requirements
--------------------
The scripts have been tested on the following system: 

Linux: Rocky Linux 8.8

GPU: NVIDIA GeForce RTX 2080 Ti

packages: python 3.8.16, pandas 1.5.3, numpy 1.23.5, pytorch 1.13.1, matplotlib 3.6.3, scikit-learn 1.2.1

installation instructions and sample usage using toy example (on CPU) 
------------------------------------------
1) create conda environment (using the provided yml file)
```
conda env create -f environment_nnlogreg.yml -n nnlogreg
conda activate nnlogreg
pip install pytorch_tabnet
pip install xgboost
```

2) train model
python NNlogreg_train.py --train_df /toydataset/normalized_training_toy_df.csv --train_labels /toydataset/normalized_training_toy_labels.csv --num_epochs 50 --save_dir \toydataset\ --batch_size 4

3) test model
python NNlogreg_test.py --test_df /toydataset/normalized_testing_toy_df.csv --test_labels /toydataset/normalized_testing_toy_labels.csv --model_path [path_to_saved_model] --batch_size 4

