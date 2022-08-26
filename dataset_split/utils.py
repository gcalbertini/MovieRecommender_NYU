
# -*- coding: utf-8 -*-
import warnings
from pyspark.sql.window import Window
from pyspark.sql import functions as fn


def readRDD(spark, dirstring, small, column_name):
    # TODO instead of manually writing this in, column_names should be read in from spark methods? A lot of repetition here
    if small == True:
        print('Using small set...')
        print('')
        dir = dirstring + '/ml-latest-small/'
        column_names = ['movies', 'ratings', 'links', 'tags']
        # Load into DataFrame
        if column_name in column_names:
            if column_name == 'movies':
                return spark.read.csv(dir + 'movies.csv', header=True, schema='movieId INT, title STRING, genres STRING'), column_name
            elif column_name == 'links':
                return spark.read.csv(dir + 'links.csv', header=True, schema='movieId INT, imdbId FLOAT, tmdbId FLOAT'), column_name
            elif column_name == 'ratings':
                return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT'), column_name
            elif column_name == 'tags':
                return spark.read.csv(dir + 'tags.csv', header=True, schema='userId INT, movieId INT, tag STRING, timestamp INT'), column_name
        else:
            warnings.warn(
                'Warning Message: Column name not found; opting for ratings')
            return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT'), 'ratings'
    else:
        print('Using large set...')
        print('')
        dir = dirstring + '/ml-latest/'
        column_names = ['movies', 'ratings', 'links',
                        'tags', 'genome-tags', 'genome-scores']
        # Load into DataFrame
        if column_name in column_names:
            if column_name == 'movies':
                return spark.read.csv(dir + 'movies.csv', header=True, schema='movieId INT, title STRING, genres STRING'), column_name
            elif column_name == 'links':
                return spark.read.csv(dir + 'links.csv', header=True, schema='movieId INT, imdbId FLOAT, tmdbId FLOAT'), column_name
            elif column_name == 'ratings':
                return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT'), column_name
            elif column_name == 'tags':
                return spark.read.csv(dir + 'tags.csv', header=True, schema='userId INT, movieId INT, tag STRING, timestamp INT'), column_name
            elif column_name == 'genome-scores':
                return spark.read.csv(dir + 'genome-scores.csv', header=True, schema='movieId INT, tagId INT, relevance STRING'), column_name
            elif column_name == 'genome-tags':
                return spark.read.csv(dir + 'genome-tags.csv', header=True, schema='tagId INT, tag STRING'), column_name
        else:
            warnings.warn(
                'Warning Message: Column name not found; opting for ratings')
            return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT'), 'ratings'


def ratings_split(rdd, train_ratio=0.6, user_ratio=0.5, seed = 15):
    windowSpec = Window.partitionBy('userId').orderBy('timestamp')

    counts = rdd.groupBy('userId').agg(fn.count('rating').alias('n_ratings'))
    rdd = rdd.join(counts, 'userId')

    ratings = rdd \
       .withColumn('row_number', fn.row_number().over(windowSpec)) \
       .withColumn('prop_idx', (fn.col('row_number') / fn.col('n_ratings')))
    ratings.show(20)

    ratings_train = ratings.filter(ratings.prop_idx <= train_ratio).drop(
        'row_number', 'n_ratings', 'prop_idx')

    ratings_val_test = ratings.filter(ratings.prop_idx > train_ratio).drop(
        'row_number', 'n_ratings', 'prop_idx')
    distinct_user_ids = ratings_val_test.select(
        'userId').distinct().alias('userId')
    user_ids_sample = distinct_user_ids.sample(
        withReplacement=False, fraction=user_ratio, seed=seed).withColumn('filter_val', fn.lit(1))

    ratings_val_test = ratings_val_test.join(
        user_ids_sample, on='userId', how='left')

    ratings_validation = ratings_val_test.filter(
        ratings_val_test.filter_val == 1).drop('filter_val')
    ratings_test = ratings_val_test.filter(
        ratings_val_test.filter_val.isNull()).drop('filter_val')

    return ratings_train, ratings_test, ratings_validation
