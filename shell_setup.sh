#!/bin/bash

export HADOOP_EXE='/usr/bin/hadoop'
export HADOOP_LIBPATH='/opt/cloudera/parcels/CDH/lib'
export HADOOP_STREAMING='hadoop-mapreduce/hadoop-streaming.jar'

alias hfs="$HADOOP_EXE fs"
alias hjs="$HADOOP_EXE jar $HADOOP_LIBPATH/$HADOOP_STREAMING"

# Creating the spark session object: --py-files directive sends the zip file to the Spark workers but does not add it to the PYTHONPATH; to add the dependencies to the PYTHONPATH to fix the ImportError add to main.py .config settings
alias spark-submit='PYSPARK_PYTHON=$(which python) spark-submit \
 --py-files dist/Group26_MovieLens-1.0.0-py3-none-any.zip \
 main.py'

module purge

#module load python/gcc/3.7.9
module load anaconda3/2020.11
module load spark/3.0.1

python build.py
