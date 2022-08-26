# Group 26: MovieLens - Basic Recommender Systems and Extensions

The following details salient features of code and the process for execution:

## Overview
- Sourcing the shell script will create Hadoop aliases, Spark Session object, load modules on the cluster environment, and run automatic building of packaging modules on wheels and zipping them up for cluster use automatically
- Run popularity baseline model selection from main routine 
- Run ALS model selection from main routine
- Write relevant csv files for post-processing on user HDFS

Let it be known that we hereby implement the requirements set on the provided project description README found in our [repo](https://github.com/nyu-big-data/final-project-group-26).

> In the final project, you will apply the tools you have learned in this class to solve a realistic,
> large-scale, applied problem. Specifically, you will build and evaluate a collaborative-filter based recommender system. Two extensions will be created for full credit. 

> In addition to all of your code, produce a final report (not to exceed 6 pages), describing your implementation, evaluation results, and extensions. Your report should clearly identify the contributions of each member of your group. If any additional software components were required in your project, your choices should be described and well motivated here.

## Additional Software Components

- [Anaconda 2020.11](https://www.anaconda.com/blog/individual-edition-2020-11) - Anaconda module providing needed libraries for running source code
- [Spark 3.0.1](https://spark.apache.org/docs/3.0.1/) - Large-scale data processing done on Apache Spark
- [NYU Peele Cluster](https://www.nyu.edu/life/information-technology/research-computing-services/high-performance-computing/high-performance-computing-nyu-it/hpc-supercomputer-clusters/peel-hadoop.html) - Peel is NYU's newest Hadoop cluster


And of course the code is found in our Github repository [here](https://github.com/nyu-big-data/final-project-group-26)!

## Background on Packaging for Nerds (Optional)

We use [setuptools](https://pypi.org/project/setuptools/) (extension for building wheels that provides the ```bdist_wheel``` setuptools command) in the project to package our custom classes and methods. For the sake of brevity here, the term package is used both as the name of a container of modules as well as the cabinet used to package it. Python’s first mainstream packaging format was the .egg file that was later replaced by the wheel (.whl), which contains all the files for a PEP 376 compatible install. Noting that *.whl file is similar to an *.egg in that it’s basically a *.zip file, we automatically scan for such file formats and rename the extension from *.whl to *.zip (see ```build.py```) just because it was discovered that the ```--py-files``` flag passes our resources to a temporary directory created on HDFS just for the lifetime of that application and, to avoid an import error, then load the files to the execution environment, like the JVM environment, using the `.config` properties in the main routine in a format (i.e. zip) that the PySpark API recognizes. To then get the dependency distribution to work with compiled extensions, first run 
```sh
python3 setup.py bdist_wheel
```
to ensure compatible binaries are included in your zip. This source distribution contains source code: this is not only Python code, but also the source code of any extension modules (usually in C or C++, but in Python here) bundled with the package. Source distributions contain a bundle of metadata files in a directory called "<package-name>.egg-info" that help with building and installing the package, but users don’t need to touch this. Note that we are also adding the minimal metadata necessary to create a package in ```setup.py``` that also looks for valid package contents. We also opt for the wheels distribution format as it lets you skip the build stage required with source distributions which thereby increases delivery and installation speed and avoids the requisite of having development tools and libraries installed on the system (i.e. user doesn't need to compile extension modules on their side if needed.) 

Since we are dealing with a server here, things get dicey so we must additionally unzip our archive on the destination node since Python will not import compiled extensions from zip files! Finally, the dependencies can now be deployed, unzipped, and included in the ```PYTHONPATH``` by including them in the terminal-side, which is put in shell script for convenience:
```
alias spark-submit='PYSPARK_PYTHON=$(which python) spark-submit \
 --py-files dist/Group26_MovieLens-0.5.0-py3-none-any.zip \
 main.py'
```

## First Time Setup
As a shortcut to this all (and avoiding the use of `requirements.txt` as we couldn't configure pip install on the Peele cluster) was to run the aforementioned all in one go by sourcing the shell script and then submitting the job. Quite intuitive, this was.

```sh
source shell_setup.sh
<distribution metadata generated here>
spark-submit
```

### Installing Distribution Packges on the Cluster: Packaging ```validated_models``` and ```dataset_split``` Modules 
We opted for a more modular approach that enforced more of a OOP approach. Modules provide a handy way to split the code into more files within a namespace; they are nothing but files containing Python code the main program can import if needed which then promotes maintainability and code re-usability. Packages are a handy way to group related modules altogether! Distribution packages we discussed prior then become versioned archive files that contain import packages, modules, and other resource files used for that project run on the cluster.

## Dataset Split
As regards the training set, for each user in the dataset, a percentage of observations (``train_ratio``) is selected to be included in the training set based on the value of the timestamp (older ratings are included in the training). The remaining data is then splitted in a way that ``user_ratio`` of the users are included in the validation set and the remaining in the test set.
We thus include a portion of the history of each user in the training set, while for the remaining observations we perfrom a user-based split, where the observations associated to ``user_ratio`` of the users fall into the validation set while the remaining go into the test set.

## Popularity Baseline
The ``PopularityBaseline`` class implements a standard popularity baseline model that gets the utility matrix containing users' ratings and computes the top most popular movies, where popularity is defined as the average rating for each movie.
 
 **Parameters:**     
 - threshold: the number of ratings a movie needs to have to be included in the training data            
 - damping: the damping factor of the model

```fit(ratings, top_n=100):```

Fit the recommender with the train rating data and computes the ``top_n`` recommendations.

**Parameters:**
- ratings: the train rating data
- top_n: the number of recommendations made by the model

**Returns:**
- results: the dataframe with the ``top_n`` recommendations in descending order (from most popular to less popular)

```evaluate(results, test_set):```

First determines the ground truth by getting the movies that the users enjoyed (rating > 2.5). It returns a ``RankingMetrics`` object to compute the preferred metrics.

**Parameters:**
- results: the dataframe with the recommendations in descending order
- test_set: the test rating data

**Returns:**
- metrics: a ``RankingMetrics`` object to compute the desired metrics

## Alternating Least Squares (ALS) 
The ``ValidateALS`` class wraps an ``ALS`` object from ``pyspark.ml.recommendation``. It performs evaluation of an ``ALS`` object and implements a standard validation process over given sets of parameters. 

**Parameters:** 
- seed: the seed to be used in the ``ALS`` class
 
```validate(ratings_train, ratings_val, top_k, rank, regParam, maxIter, ColdStartStrategy, metric, verbose):```
 
Performs validation of the ALS model over the given sets of parameter, outputting the best performing model.
 
**Parameters:**
- ratings_train: the train rating data
- ratings_val: the validation rating data
- top_k: the number of recommendations for each user
- rank: the list of rank values for the validation of the ALS
- regParam: the list of regression parameters for the validation of the ALS
- maxIter: the list of maximum iterations for the validation of the ALS
- ColdStartStrategy: the paramater to implement the cold start strategy in the ALS, default set to "nan"
- metric: the type of metric used to evaluate the models perfromances
- verbose: set to True to print the intermediate results of the validation


**Returns:**
- self.model: the model fitted on the best parameter configuration

```evaluate(ratings_test, top_k, metricName):```

To evaluate the ALS models fitted during validation. Given a test set computes the ground truth by selecting the movies with rating > 2.5, and compares them to the ``top_k`` user specific recomendations by using a ``RankingEvaluator`` object.

**Parameters:**
- ratings_test: the test rating data
- top_k: the number of recommendations to produce for each user
- metricName: the metric used to evaluate the model

**Returns:**
- score: the score for the fitted ALS model on the given test rating data

## Extension 1: Comparison to Single Machine Implementation
The perfromances (accuracy and time to fit) obtained from the Spark parallel ALS runned on the cluster are compared to the ones obtained from the ALS runned on a single machine. To implement the ALS on the single machine the [lenskit](https://lkpy.readthedocs.io/en/stable/index.html) library is used. 

We implement the function ``SingleMachineValidation`` to perfrom validation of lenskit's ALS algorithm on the same sets of parameters used in the Spark version for the sake of comparison. 

```SingleMachineImplementation(X_train, X_val, rank_vals, regParam_vals, maxIter_vals, metric_val, k_val, verbose, size):```

**Parameters:**
- X_train: the train rating data
- X_val: the validation rating data
- rank_vals: the list of rank values for the validation of the ALS
- regParam_vals: the list of regression parameters for the validation of the ALS
- maxIter_vals: the list of maximum iterations for the validation of the ALS
- metric_val: the type of metric used to evaluate the models perfromances
- k_val: the number of recommendations for each user
- verbose: True if needed all the intermediate results
- size: the size of the dataset used (either "small" or "big")

**Returns:**
- best_model: the best model configuartions obtained from validation
- best_fittable: the fitted best model used to perform recommendations

## Extension 2: Fast Search
Using the user and item factor matrices of the ALS model with the best hyperparameter configurations, an implementation accelerated search at query time is provided with the help of a spatial data structure. The fast search implementation has then been compared to the brute-force method. The [annoy](https://github.com/spotify/annoy) library has been used to implement the fast search method.

### Brute Force
The ``BruteForce`` class implements recommendations using the estimated user factors and item factors by brute-force, i.e. by computing the inner products of the query (user latent factors) with all the item factor representations and getting the items with the `k` highest inner products, where `k` is the number of recommendations to provide.

**Parameters:**
- user_factors_array: the array containing the estimated user factors
- item_factors_array: the array containing the estimated item factors
- user_ids: the array containing the user identifiers
- item_ids: the array containing the item identifiers

```query(user_factors, top_k):```

Generates `top_k` item recommendations for the given query `user_factors` array.

**Parameters:**
- user_factors: user factor query array
- top_k: number of recommendations to generate

**Returns:**
- recommendations: list of item positional indices in the item_factors_array
- query_time: time to generate the recommendations

```recommendations(n_queries, top_k):```

Generates `top_k` item recommendations for the top `n_queries` rows in the user_factors_array. The recommendations for each of the given queries, in terms of the user and item identifiers contained in user_ids and item_ids, are stored as a class attribute. Also the times to generate each query are stored as a class attribute.

**Parameters:**
- n_queries: number of top rows in user_factors_array to generate recommendations for
- top_k: number of recommendations to generate

### Fast Search with Annoy
The ``AnnoyFS`` class implements recommendations using the estimated user factors and item factors using the Annoy fast search method.

**Parameters:**
- user_factors_array: the array containing the estimated user factors
- item_factors_array: the array containing the estimated item factors
- user_ids: the array containing the user identifiers
- item_ids: the array containing the item identifiers

```build_index(n_trees, metric):```

Builds an Annoy index with the specified `n_trees` and `metric` parameters. The index is stored as a class attribute. Also the time to build the index is stored as a class attribute.

**Parameters:**
- n_trees: number of trees that are built in the Annoy index
- metric: metric used to define distance in the space

```query(user_factors, top_k):```

Generates `top_k` item recommendations for the given query `user_factors` array.

**Parameters:**
- user_factors: user factor query array
- search_k: number of total nodes that will be searched in the Annoy index
- top_k: number of recommendations to generate

**Returns:**
- recommendations: list of item positional indices in the item_factors_array
- query_time: time to generate the recommendations

```recommendations(n_queries, top_k):```

Generates `top_k` item recommendations for the top `n_queries` rows in the user_factors_array. The recommendations for each of the given queries, in terms of the user and item identifiers contained in user_ids and item_ids, are stored as a class attribute. Also the times to generate each query are stored as a class attribute.

**Parameters:**
- n_queries: number of top rows in user_factors_array to generate recommendations for
- search_k: number of total nodes that will be searched in the Annoy index
- top_k: number of recommendations to generate

### Helper functions

```compute_recall(users_recommendations_fs, users_recommendations_bf):```

Computes recall of each user's recommendations generated with fast search with respect to the "true" recommendations generated by brute force.

**Parameters:**
- users_recommendations_fs: dictionary containing list of recommendations (value) for each user (key) generated with approximate nearest neighbor method
- users_recommendations_bf: dictionary containing list of recommendations (value) for each user (key) generated by brute force (true nearest neighbours)

**Returns:**
- recall_values: dictionary containing recall value (value) for each user (key)


```comparison(user_factors_array, item_factors_array, user_ids, item_ids, n_trees_list, search_k_list, n_queries, top_k):```

Compares recommendations by brute force and with the Annoy method, for different configurations of Annoy index building/query generation, in terms of queries per second and average recall over all queries. The scores for each method/configuration are printed.

**Parameters:**
- user_factors_array: the array containing the estimated user factors
- item_factors_array: the array containing the estimated item factors
- user_ids: the array containing the user identifiers
- item_ids: the array containing the item identifiers
- n_trees_list: list of n_trees values for Annoy index building
- search_k_list: list of search_k values for Annoy query generation
- n_queries: number of top rows in user_factors_array to generate recommendations for
- top_k: number of recommendations to generate

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [wheel]: <https://wheel.readthedocs.io/en/stable/>
   [Python setup tools]: <https://grimoire.carcano.ch/blog/python-setup-tools/>
   [PyFiles on Spark]: <https://newbedev.com/i-can-t-seem-to-get-py-files-on-spark-to-work/>

