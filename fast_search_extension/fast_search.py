#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
from annoy import AnnoyIndex


# Produce recommendations by brute force with heap queue algorithm to get top k items

class BruteForce():
    
    def __init__(self, user_factors_array, item_factors_array, user_ids, item_ids):
        self.user_factors_array = user_factors_array
        self.item_factors_array = item_factors_array
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.users_recommendations = None
        self.query_times = None
    
    def query(self, user_factors, top_k=100):
        start_time = time.time()
        inner_products = []
        for j in range(len(self.item_factors_array)):
            inner_product = np.dot(user_factors, self.item_factors_array[j, :])
            inner_products.append(inner_product)
        k_largest = heapq.nlargest(top_k, enumerate(inner_products), key=lambda x: x[1])
        recommendations = list(list(zip(*k_largest))[0])
        end_time = time.time()
        query_time = end_time - start_time
        return recommendations, query_time
    
    def recommendations(self, n_queries = 100, top_k = 100):
        users_recommendations = {}
        query_times = []
        for i in range(n_queries):
            recommendations, query_time = self.query(self.user_factors_array[i, :], top_k=top_k)
            users_recommendations[self.user_ids[i]] = list(self.item_ids[recommendations])
            query_times.append(query_time)
        self.users_recommendations = users_recommendations
        self.query_times = query_times
 

# Produce recommendations with Annoy fast search method

class AnnoyFS():
    
    def __init__(self, user_factors_array, item_factors_array, user_ids, item_ids):
        self.user_factors_array = user_factors_array
        self.item_factors_array = item_factors_array
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.index = None
        self.index_time = None
        self.users_recommendations = None
        self.query_times = None
    
    def build_index(self, n_trees = 1000, metric='dot'):
        f = self.user_factors_array.shape[1]
        self.index = AnnoyIndex(f,  metric)
        for i in range(len(self.item_factors_array)):
            self.index.add_item(i, item_factors_array[i, :])
        index_start_time = time.time()
        self.index.build(n_trees, n_jobs=-1)
        index_end_time = time.time()
        self.index_time = index_end_time - index_start_time
        
    
    def query(self, user_factors, search_k = 1000, top_k=100):
        start_time = time.time()
        recommendations = self.index.get_nns_by_vector(user_factors, top_k,
                                              search_k,
                                              include_distances=False)
        end_time = time.time()
        query_time = end_time - start_time
        return recommendations, query_time
    
    def recommendations(self, n_queries = 100, search_k = 1000, top_k = 100):
        users_recommendations = {}
        query_times = []
        for i in range(n_queries):
            recommendations, query_time = self.query(self.user_factors_array[i, :], search_k=search_k, top_k=top_k)
            users_recommendations[self.user_ids[i]] = list(self.item_ids[recommendations])
            query_times.append(query_time)
        self.users_recommendations = users_recommendations
        self.query_times = query_times
   

# Function to compute recall over all the recommendations produced with fast search

def compute_recall(users_recommendations_fs, users_recommendations_bf):
    recall_values = {}
    
    for user in users_recommendations_bf.keys():
        recommendations_fs = users_recommendations_fs[user]
        recommendations_bf = users_recommendations_bf[user]
        recall = len(set(recommendations_fs) & set(recommendations_bf)) / len(recommendations_fs)
        recall_values[user] = recall
    
    return recall_values


def comparison(user_factors_array, item_factors_array, user_ids, item_ids, n_trees_list, search_k_list, n_queries=100, top_k=100):
    
    print(f'Producing {top_k} recommendations for {n_queries} users by brute force')
    print()
    bf = BruteForce(user_factors_array, item_factors_array, user_ids, item_ids)
    bf.recommendations(n_queries=n_queries, top_k=top_k)
    queries_per_sec_bf = n_queries / sum(bf.query_times)
    print(f'Queries per second = {queries_per_sec_bf}')
    print(f'Average recall over all queries = 1.0')
    print()
    print('------------------------------------------------------------------')
    print()
    
    print(f'Producing {top_k} recommendations for {n_queries} users with Annoy fast search method (inner product distance)')
    print()

    for n_trees in n_trees_list:
        
        fs = AnnoyFS(user_factors_array, item_factors_array, user_ids, item_ids)
        fs.build_index(n_trees, metric='dot')
        
        for search_k in search_k_list:
            
            print(f'Annoy configuration: n_trees = {n_trees}, search_k = {search_k}')
            
            fs.recommendations(n_queries=n_queries, search_k=search_k, top_k=top_k)
            queries_per_sec_fs = n_queries / sum(fs.query_times)
            avg_recall = np.mean(list(compute_recall(fs.users_recommendations, bf.users_recommendations).values()))
            
            print(f'Time to build index: {fs.index_time} seconds')
            print(f'Queries per second = {queries_per_sec_fs}')
            print(f'Average recall over all queries = {avg_recall}')
            print()
    

# import and preprocessing of item factors

item_factors = pd.read_csv('data/item_factors.csv',
                            header=None,
                            names=['movieId', 'features'])
item_factors['features'] = item_factors['features'].apply(lambda x: list(map(float, x.split(','))))
item_factors = pd.concat([pd.DataFrame(item_factors['movieId']), pd.DataFrame(item_factors['features'].to_list())], axis=1)
item_factors = item_factors.sort_values(by=['movieId']).reset_index(drop=True)

item_factors_array = np.array(item_factors.iloc[:, 1:])


# import and preprocessing of user factors

user_factors = pd.read_csv('data/user_factors.csv',
                            header=None,
                            names=['userId', 'features'])
user_factors['features'] = user_factors['features'].apply(lambda x: list(map(float, x.split(','))))
user_factors = pd.concat([pd.DataFrame(user_factors['userId']), pd.DataFrame(user_factors['features'].to_list())], axis=1)
user_factors = user_factors.sort_values(by=['userId']).reset_index(drop=True)

user_factors_array = np.array(user_factors.iloc[:, 1:])

n_trees_list = [100, 500, 1000]
search_k_list = [100, 1000, 10000, 50000, 100000]


comparison(user_factors_array=user_factors_array, item_factors_array=item_factors_array,
            user_ids=user_factors['userId'], item_ids=item_factors['movieId'], n_queries=100,
            top_k=100, n_trees_list=n_trees_list, search_k_list=search_k_list)
    
    


