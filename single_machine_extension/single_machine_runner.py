#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from single_validation import SingleMachineValidation
from lenskit.algorithms import Recommender, als
from lenskit import topn, batch

X_train = pd.read_csv('../../small/ratings_train.csv', names=['user', 'item', 'rating', 'timestamp'])
X_val = pd.read_csv('../../small/ratings_test.csv', names=['user', 'item', 'rating', 'timestamp'])
X_test = pd.read_csv('../../small/ratings_test.csv', names=['user', 'item', 'rating', 'timestamp'])
#We select only the movies viewed by the user, which have a rating higher than 2.5. 
#In this way we interpret a right recommendation the one with 
X_val = X_val.loc[X_val['rating']>2.5]
X_test = X_test.loc[X_test['rating']>2.5]

rank_values = [20, 30, 40, 50]
regParam_values = [0.001, 0.01, 0.1, 1]
maxIter_values = [10, 15, 20, 25, 30]
top_k = 100

print("Tuning hyperparameters based on Mean Average Precision")

best_model, best_fittable = SingleMachineValidation(X_train, X_val, rank_values, regParam_values, maxIter_values, k_val = top_k)

print("Evaluating best Popularity baseline model")
users = X_test.user.unique()
recs = batch.recommend(best_fittable, users, top_k)
rla = topn.RecListAnalysis()
rla.add_metric(topn.ndcg)
rla.add_metric(topn.precision)
results = rla.compute(recs, X_test)
NDCG = results.ndcg.mean()
MAP = results.precision.mean()
print("MAP@100 on test set: ", MAP)
print("NCDG@100 on test set: ", NDCG)
