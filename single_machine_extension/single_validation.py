#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from lenskit.algorithms import Recommender, als
from lenskit import topn, batch, util

def SingleMachineValidation(X_train, X_val, rank_vals = [10], regParam_vals = [0.1], maxIter_vals = [10], metric_val= 'meanAveragePrecision',\
                                 k_val = 100, verbose = True, size = 'small'):
    tot_times = []
    best_score = 0
    best_model = None
    best_fittable = None
    for rank in rank_vals:
        for regParam in regParam_vals:
            for maxIter in maxIter_vals:
                if verbose:
                    print("Fitting ALS model on a single machine given parameters: Rank: {rank} | regParam: {regParam} | maxIter: {maxIter} ".format(
                    rank=rank, regParam=regParam, maxIter=maxIter))
                model = als.BiasedMF(features = rank, iterations = maxIter, reg = regParam, bias = False)
                fittable = util.clone(model)
                fittable = Recommender.adapt(fittable)
                start = time.time()
                fittable.fit(X_train)
                end = time.time()
                tot_times.append(end-start)
                users = X_val.user.unique()
                recs = batch.recommend(fittable, users, k_val)
                if metric_val == 'meanAveragePrecision':
                    rla = topn.RecListAnalysis()
                    rla.add_metric(topn.precision)
                    results = rla.compute(recs, X_val)
                    val_score = results.precision.mean()
                    metric_val = 'MAP'
                elif metric_val == 'ndcg':
                    rla = topn.RecListAnalysis()
                    rla.add_metric(topn.ndcg)
                    results = rla.compute(recs, X_val)
                    val_score = results.ndcg.mean()
                    metric_val = 'NDCG'
                else:
                    raise Exception('metric not supported')
            
                if verbose:
                    print('Score for ALS model given these parameters using {m}@{k}: {s}'.format(m = metric_val, k = k_val, s = val_score))
                    print('------------------------------------------------------------------------------------------------------------------------------')
                
                if val_score > best_score:
                    best_score = val_score
                    best_model = model
                    best_fittable = fittable
                    
   
    print('==============================================================================================================================')
    print("Best ALS model given parameters using {m}@{k}: Rank: {rank} | regParam: {regParam} | maxIter: {maxIter}".format(
        rank=best_model.features, regParam=best_model.regularization, maxIter=best_model.iterations, m = metric_val, k = k_val))
    print("Best ALS model score given these parameters: ", best_score)
    print("Average model fitting time for the {size} dataset: {time}".format(size = size, time = np.mean(tot_times)))
    return best_model, best_fittable
        
