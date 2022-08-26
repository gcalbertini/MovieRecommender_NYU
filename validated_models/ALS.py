# -*- coding: utf-8 -*-

# https://www.timlrx.com/blog/creating-a-custom-cross-validation-function-in-pyspark

import pyspark.sql.functions as fn
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as fn
from pyspark.ml.evaluation import RankingEvaluator
import time
import numpy as np

class ValidatedALS():

    def __init__(self, seed=0):

        self.fitted = False
        self.seed = seed
        self.model = None
        self.predsAndlabels = None
        self.score = 0

    def validate(self, ratings_train, ratings_val, top_k=100, rank=[10], regParam=[0.1], maxIter=[10], coldStartStrategy="nan", verbose=True, metric='meanAveragePrecision'):
        
        self.k = top_k
        self.coldStartStrategy = coldStartStrategy
        
        best_predsAndlabels = None
        best_score = 0
        best_model = None
        best_rank = None
        best_maxIter = None
        best_regParam = None

        als = ALS(userCol="userId", itemCol="movieId",
                  ratingCol="rating", seed=self.seed)

        tot_times = []
        
        for reg in regParam:
            for r in rank:
                for i in maxIter:
                    als.setMaxIter(i)
                    als.setRank(r)
                    als.setRegParam(reg)

                    if verbose:
                        print("Fitting ALS model given parameters: Rank {r} | MaxIter: {i} | RegParam: {reg} | ColdStartStrategy: {strat} |".format(
                           r=r, i=i, reg=reg, strat=self.coldStartStrategy))
                    
                    start = time.time()
                    self.model = als.fit(ratings_train)
                    end = time.time()
                    tot_time = end - start
                    tot_times.append(tot_time)
                    self.fitted = True

                    self.score = self.evaluate(ratings_val, top_k, metric)

                    if verbose:
                        if metric == 'meanAveragePrecision':
                            print('Score for ALS model given these parameters using MAP@{k}: {s}'.format(k = top_k, s = self.score))
                            print("Model fitting time for the given dataset: ",tot_time)
                            print('------------------------------------------------------------------------------------------------------------------------------')     
                        elif metric == 'ndcgAtK':
                            print('Score for ALS model given these parameters using NCDG@{k}: {s}'.format(k = top_k, s = self.score))
                            print("Model fitting time for the given dataset: ",tot_time)
                            print('------------------------------------------------------------------------------------------------------------------------------')     
                        

                    if self.score > best_score:
                        best_model = self.model
                        best_score = self.score
                        best_predsAndlabels = self.predsAndlabels
                        best_rank = r
                        best_maxIter = i
                        best_regParam = reg

        print('==============================================================================================================================')
        if metric == 'meanAveragePrecision':
            print("Best ALS model given parameters using MAP@{k}: Rank {r} | MaxIter: {i} | RegParam: {reg} | ColdStartStrategy: {strat} |".format(
                k=top_k, r= best_rank, i=best_maxIter, reg=best_regParam, strat=self.coldStartStrategy))
            print("Best ALS model score given these parameters: ", best_score)
        else: 
            print("Best ALS model given parameters using NCDG@{k}: Rank {r} | MaxIter: {i} | RegParam: {reg} | ColdStartStrategy: {strat} |".format(
                k=top_k, r= best_rank, i=best_maxIter, reg=best_regParam, strat=self.coldStartStrategy))
            print("Best ALS model score given these parameters: ", best_score)
        print("Average model fitting time for the given dataset: ", np.mean(tot_times))

        self.model = best_model
        self.score = best_score
        self.predsAndlabels = best_predsAndlabels
        self.rank = best_rank
        self.maxIter = best_maxIter
        self.regParam = best_regParam
        
        return self.model

    def evaluate(self, ratings_test, top_k, metricName='meanAveragePrecision'):

        data = ratings_test.filter(ratings_test.rating > 2.5).drop(
            'rating', 'timestamp')
        UserMovies = data.groupBy('userId').agg(
            fn.collect_list('movieId').alias('label'))

        # Generate top k movie recommendations for each user
        df_recs = self.model.recommendForAllUsers(top_k).withColumn(
            'recommendations', fn.explode((fn.col('recommendations'))))
        df_recs = df_recs.withColumn('recommendations', df_recs.recommendations.getItem(
            'movieId')).groupBy('userId').agg(fn.collect_list('recommendations').alias('recommendations'))
        
        self.predsAndlabels = df_recs.join(UserMovies, 'userId').select(fn.col('recommendations').cast(
            'array<double>').alias('recommendations'), fn.col('label').cast('array<double>').alias('label'))
        
        if metricName == 'meanAveragePrecision':
            evaluator = RankingEvaluator(metricName=metricName).setPredictionCol('recommendations')
        elif metricName == 'ndcgAtK':
            evaluator = RankingEvaluator(metricName=metricName).setPredictionCol('recommendations').setK(top_k)

        return evaluator.evaluate(self.predsAndlabels)
