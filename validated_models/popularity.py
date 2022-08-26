#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyspark.sql.functions as fn
from pyspark.mllib.evaluation import RankingMetrics


class PopularityBaseline():

    def __init__(self, threshold=None, damping=0):

        self.threshold = threshold
        self.damping = damping
        self.fitted = False
        self.results = None

    def __repr__(self):
        return 'PopularityBaseline(threshold = ' + str(self.threshold) + ', damping = ' + str(self.damping) + ', fitted = ' + str(self.fitted) + ')'

    def fit(self, ratings, top_n=100):
        ratings = ratings.groupBy('movieId').agg(fn.sum('rating').alias('tot_rating'), fn.count('userId').alias('counts')) \
            .withColumn('counts', fn.col('counts') + self.damping)
        if self.threshold is not None:
            ratings = ratings.filter(ratings.counts >= self.threshold)
        results = ratings.withColumn('popularity', fn.col(
            'tot_rating') / fn.col('counts')).drop('tot', 'counts').orderBy(fn.desc('popularity'))
        results = results.limit(top_n).select('movieId')
        self.fitted = True
        self.results = results
        return results

    def evaluate(self, results, test_set):
        data = test_set.filter(test_set.rating > 2.5).drop(
            'rating', 'timestap')

        results = results.withColumn('id', fn.lit(1)).groupBy(
            'id').agg(fn.collect_list('movieId').alias('top_movies'))

        UserMovies = data.groupBy('userId').agg(
            fn.collect_list('movieId').alias('liked_movies'))

        UserMovies = UserMovies.withColumn('id', fn.lit(1))

        UserMovies = UserMovies.join(results, 'id').select(
            'top_movies', 'liked_movies')

        predictionAndLabels = UserMovies.rdd.map(tuple)

        metrics = RankingMetrics(predictionAndLabels)

        return metrics


# evaluation done by using Spark RankingMetrics which takes as in put an rdd where each user is a row and for each user have list of recommended movies
# and list of ground truth movies. So for each user get the ground truth as those movies watched with rating > 2.5
