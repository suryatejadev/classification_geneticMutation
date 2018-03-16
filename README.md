Project Title	: Clinically Actionable Genetic Mutation Classification for Cancer Treatment
Duration		: Aug 17 - Dec 17

Description: This project is an individual project done in the course 'COMPSCI 682 Neural Networks: A modern introduction'. It is a NIPS 2017 Kaggle competition 'Personalized Medicine: Redefining Cancer Treatment'. Further details of this competition can be found at https://www.kaggle.com/c/msk-redefining-cancer-treatment.

Problem Statement: The objective of this project is to build an automated multi-class classification system for the prediction of the type of cancer caused by the genetic mutations. The classification is based on clinical text evidence, which in the form of biomedical publications. Experiments are conducted with several word embeddings techniques, data imbalance mitigation methods and classification models. The model constructed using a combination of Word2Vec word embedding, label encoding, data expansion using oversampling, and an artificial neural network using batch normalization, dropout and Adam optimizer, gave the most balanced confusion matrix, and a multi-class log loss of 1.688.

Files:
1. features/	: to store the word embeddings.
2. func/		: contains the utility files.
3. misc/		: contains miscellaneous files.
4. output/		: folder where the output is stored
5. extractFeat.py:	main code for extracting word embeddings under various models like word2vec, doc2vec, etc.
5. train_classifiers.py	: main code for classification of the word embeddings using ANN, randomforest, SVM, XGBoost.
6. train_cnn.py	: main code for classification of the word embeddings of word2vec using 1D-CNN.
7. train_lstm.py: main code for classification of the word embeddings of word2vec using LSTM.





