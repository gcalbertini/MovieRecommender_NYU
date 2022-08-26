# -*- coding: utf-8 -*-

import getpass
from validated_models.ALS import ValidatedALS
from dataset_split.utils import readRDD
from validated_models.popularity import PopularityBaseline
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as fn
from pyspark.sql import SparkSession
import dataset_split



def main(spark, in_path, out_path):
    
    print('LOADING....')
    print('')

    print('Splitting the ratings dataset into training, validation and test set')
    ratings_train, ratings_test, ratings_validation = dataset_split.ratingsSplit(
        spark, in_path, small=False, column_name='ratings', train_ratio=0.8, user_ratio=0.5)
    ratings_train.show()

    print("Distinct movies: ", ratings_train.select(
        "movieId").distinct().count())
    print("Distinct users: ", ratings_train.select("userId").distinct().count())
    print("Total number of ratings: ", ratings_train.count())

    X_train, X_test, X_val = ratings_train.drop('timestamp'), ratings_test.drop(
        'timestamp'), ratings_validation.drop('timestamp')
    
    ratings_per_user = ratings_train.groupby('userId').agg({"rating":"count"})
    ratings_per_user.describe().show()

    ratings_per_movie = ratings_train.groupby('movieId').agg({"rating":"count"})
    ratings_per_movie.describe().show()

    print("Training data size : ", X_train.count())
    print("Validation data size : ", X_val.count())
    print("Test data size : ", X_test.count())
    print("Distinct users in Training set : ", X_train[["userId"]].distinct().count())
    print("Distinct users in Test set : ", X_test[["userId"]].distinct().count())
    print("Distinct users in Validation set: ", X_val[["userId"]].distinct().count())
    
    print("Fitting Popularity baseline model")
    print("Tuning hyperparameters based on Mean Average Precision")
    damping_values = [10, 20, 30, 40, 50]
    best_score = 0
    best_baseline_model = None
    for damping in damping_values:
        baseline = PopularityBaseline(damping = damping)
        baseline.fit(X_train)
        baseline_metrics_val = baseline.evaluate(baseline.results, X_val)
        val_score = baseline_metrics_val.meanAveragePrecision

        print("Fitting popularity baseline given parameters: Damping {d}".format(d=damping))
        print('Score for popularity baseline given these parameters using MAP@100: ', val_score)
        print('------------------------------------------------------------------------------------------------------------------------------')     

        if val_score > best_score:
            best_score = val_score
            best_baseline_model = baseline

    print('Best Popularity baseline model: ', best_baseline_model)
    print("Evaluating best Popularity baseline model")
    baseline_metrics_train = baseline.evaluate(best_baseline_model.results, X_train)
    baseline_metrics_test = baseline.evaluate(best_baseline_model.results, X_test)
    print("MAP@100 on training set: ", baseline_metrics_train.meanAveragePrecision)
    print("MAP@100 on test set: ", baseline_metrics_test.meanAveragePrecision)
    print("NCDG@100 on training set: ", baseline_metrics_train.ndcgAt(100))
    print("NCDG@100 on test set: ", baseline_metrics_test.ndcgAt(100))
 
    best_als_model = ValidatedALS(seed=0)
    best_als_model.validate(ratings_train=X_train, ratings_val=X_val, rank=[20, 30, 40, 50, 60],
                            regParam=[0.001, 0.01, 0.1, 1], maxIter=[25, 30, 35, 40, 45])
    
    print("Evaluating best ALS model")
    print("MAP@100 on training set: ", best_als_model.evaluate(ratings_test=X_train, top_k=100, metricName='meanAveragePrecision'))
    print("MAP@100 on test set: ", best_als_model.evaluate(ratings_test=X_test, top_k=100, metricName='meanAveragePrecision'))
    print("NCDG@100 on training set: ", best_als_model.evaluate(ratings_test=X_train, top_k=100, metricName='ndcgAtK'))
    print("NCDG@100 on test set: ", best_als_model.evaluate(ratings_test=X_test, top_k=100, metricName='ndcgAtK'))
    
    best_user_factors = best_als_model.model.userFactors
    best_item_factors = best_als_model.model.itemFactors
    
    print()
    print('Exporting User and Item factors of best ALS model into CSV')
    
    best_user_factors.withColumn("features", best_user_factors.features.cast("array<string>"))\
        .withColumn("features", fn.concat_ws(",",fn.col("features")))\
        .repartition(1).write.csv(out_path + '/final_model_results/user_factors.csv', mode='overwrite')
        
    best_item_factors.withColumn("features", best_item_factors.features.cast("array<string>"))\
        .withColumn("features", fn.concat_ws(",",fn.col("features")))\
        .repartition(1).write.csv(out_path + '/final_model_results/item_factors.csv', mode='overwrite')


# Only enter this block if we're in main
if __name__ == "__main__":

    spark = SparkSession.builder.appName('project')\
        .config('spark.submit.pyFiles', 'Group26_MovieLens-1.0.0-py3-none-any.zip')\
        .config('spark.shuffle.useOldFetchProtocol', 'true')\
        .config('spark.shuffle.service.enabled', 'true')\
        .config('dynamicAllocation.enabled', 'true')\
        .config('spark.task.maxFailures', '2')\
        .getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()

    # Get path of ratings file
    in_path = f'hdfs:/user/{netID}'

    # Get destination directory of training, validation and test set files
    out_path = f'hdfs:/user/{netID}'

    # Call our main routine
    main(spark, in_path, out_path)
