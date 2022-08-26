# -*- coding: utf-8 -*-
import sys
from . import utils


def ratingsSplit(spark, dirstring, small, column_name, train_ratio=0.6, user_ratio=0.5, seed = 15):
    # TODO you can do error checking for prop limits here if needed, i.e if lower greater than upper, is valid, etc
    rdd, verified_column_name = utils.readRDD(
        spark, dirstring, small, column_name)

    # TODO for referenc for now
    print('Printing schema')
    rdd.printSchema()

   # TODO implement for others?
    if verified_column_name == 'ratings':
        return utils.ratings_split(rdd, train_ratio, user_ratio)
    else:
        print('Something wrong??')
        sys.exit('WTF is this')


# https://docs.python-guide.org/writing/structure/
