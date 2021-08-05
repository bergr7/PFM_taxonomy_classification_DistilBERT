# Taxonomy classification of events with DistilBERT
Master in machine learning, deep learning and artificial intelligence with UCAM and Big data international campus.

Master thesis about applied NLP in a commecial setting.
_____

This repository comes along with the master's thesis submitted where the approach to this task and decisiones made are explained in detail.

The master's thesis can be found in the following link (Note that it is written in Spanish):

https://drive.google.com/drive/folders/1hmu_BwEQtaYrqGJsh96pzNNgFhmuNdzG?usp=sharing

## Objective
_______
The objective of this master's thesis is to build a taxonomy classifier, i.e., a model that processes the description of an event using natural language processing techniques and classifies it by taxonomy sub-taxonomy.

The state-of-the-art model `DistilBERT` was used for this purpose.

The taxonomy classifier was deployed using the recent tool for deploying `Pytorch` models into production `TorchServe`.

## Data
_______
The data used for training are property of [Smartvel](https://www.smartvel.com) and, therefore, have not been uploaded to the public repository.

The size of the dataset is 60,000 events with title, description, taxonomy and subtaxonomy.

For training and evaluation, the data were partitioned according to the following proportions:

- 70% Training
- 15% Validation or hold-out
- 15% Testing


## Models
________
The weights of the trained models are publicly available and can be downloaded at the link below:

https://drive.google.com/drive/folders/1LOnXoH2iCOBtD9of17uqa4PTFTQ2h9eX?usp=sharing

## Results
_________

The taxonomy classifier achieved an accuracy of **79.4%** on the validation set and **78.8%** on the testing set.

Confusion matrices:

![image](img/confusion_matrix_taxonomy_val.png)

![image](img/confusion_matrix_taxonomy.png)

The subtaxonomy classifier for taxonomy 173 achieved an accuracy of **78.7%** on the validation set and **77.9%** on the test set.

Confusion matrices:

![image](img/confusion_matrix_subtaxonomy_val.png)

![image](img/confusion_matrix_subtaxonomy.png)

Smartvel achieves a sligthly better accuracies on this task but they use BERT models, more data and of higher quality. Hence, the trained models were of use in an industrial setting since they are faster, smaller, cheaper and still provide a good performance on the classification tasks.

## Model deployment with TorchServe
_________

The taxonomy classifier model was deployed with `TorchServe`.

[Deployment with TorchServe](TorchServe/)

## Virtual environment
_______

Two virtual environments were created for the project utilizing the anaconda distribution, one for development and the other for deployment with `TorchServe`. To reproduce the results, the virtual environment can be installed from the following lines of code from the following lines of code:

- macOS:

`$ conda create --name taxonomy_distilbert --file requirements_distilbert.txt`

- Windows o Linux:

`$ conda env create -f environment_distilbert.yml`