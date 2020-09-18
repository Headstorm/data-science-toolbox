#==============================================================
# OBJECTIVE: Recommend product(s) to customers using 
#            Collaborative filtering method
#==============================================================
import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

from surprise import accuracy
from collections import defaultdict

import os
def select_cf_model(algorithms=[SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]):
    #=========================Create automated context to pick best CF model========================
    benchmark = []
    algos = []
    # Iterate over all algorithms
    for algorithm in algorithms:
        # Perform cross validation
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
        algos = algos +[algorithm]
        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)
        
    out = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')  
    return out,algos


def get_generic_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def get_Iu(uid):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
    
if __name__=="__main__":
    #==============Read in the dataset=============================================
    #  data file requirements
    #   column names:  UserId,ProductId,Rating,Timestamp"
    #   column types:  varchar, varchar,float,long
    #
    path = "/Users/aokossi/Documents/ds-projects/datasets"
    amazon_dataset = "ratings_Beauty.csv" ##UserId	ProductId	Rating	Timestamp"
    


    # Read data 
    df1 = pd.read_csv(path+'/'+amazon_dataset)
    df1['Rating'] = df1['Rating'].astype(float)
    print('Dataset 1 shape: {}'.format(df1.shape))
    print('-Dataset examples-')
    print(df1.head(10))

    df = df1.copy()

    df.index = np.arange(0,len(df))

    ### Quick EDA
    p = df.groupby('Rating')['Rating'].agg(['count'])
    print("Rating frequency count")
    print(p)
    # get product count
    product_count = df.isnull().sum()[1]
    print("missing products frequency count")
    print(product_count)
    # get customer count
    cust_count = df['UserId'].nunique() - product_count
    print("User frequency count count")
    print(cust_count)
    # get rating count
    rating_count = df['UserId'].count() - product_count
    print("Rating count")
    print(rating_count)
    # =============Some cleaning===============================


    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()

    # remove those product ID rows
    df = df[pd.notnull(df['Rating'])] 

    # create numerical user and prod index  === not needed in the algorithm used here
    prod_dimension = pd.DataFrame({'ProductId':df['ProductId'].drop_duplicates()})
    prod_dimension['prod_id'] = range(len(prod_dimension)) 
    usr_dimension = pd.DataFrame({'UserId':df['UserId'].drop_duplicates()})
    usr_dimension['user_id'] = range(len(usr_dimension))

    usr_dimension['user_id'] = usr_dimension['user_id'] + 1
    prod_dimension['prod_id'] = prod_dimension['prod_id'] +1
    df = pd.merge(df,usr_dimension,on='UserId',how="inner")
    df = pd.merge(df,prod_dimension,on='ProductId',how="inner")
    #=================================================================

    #========read data into Surprise package==========================
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Rating']], reader)

    #============Experiment with a few collaborative filtering models and pick best=======
    out,algos = select_cf_model(algorithms=[SVD(), SVDpp()])
    out = out.reset_index()
    best_algo_name = out['Algorithm'].values[0]
    best_algo_idx = out[['test_rmse']].idxmin() 
    #========================train test split================================

    trainset, testset = train_test_split(data, test_size=0.25)
    print("===optimal model index===")
    print(best_algo_idx)
    print("=========================")
    algo =  algos[best_algo_idx[0]]
    predictions = algo.fit(trainset).test(testset)
    print("Model Performance RMSE :"+ str(accuracy.rmse(predictions)))

    df_pred = pd.DataFrame(predictions, columns=['UserId', 'ProductId', 'Rating', 'est', 'details'])
    df_pred['Countproducts_Rated_by_user'] = df_pred.UserId.apply(get_Iu)
    df_pred['CountUsers_Rated_Products'] = df_pred.ProductId.apply(get_Ui)
    df_pred['err'] = abs(df_pred.est - df_pred.Rating)
    best_predictions = df_pred.sort_values(by='err')[:10]
    worst_predictions = df_pred.sort_values(by='err')[-10:]
    print("best pedictors")
    print(best_predictions)

    print("best worst")
    print(worst_predictions)
    # ====================== scoring


    trainsetfull = data.build_full_trainset()
    algo.fit(trainsetfull)

    #  - Recommendation use case 1: for a given user id = 'A39HTATAQ9V7YF' and product id = '0205616461'
    #                              is it worth recommending the product to the user?
    #                               if the predicted rating is near 5 then yes otherwise no

    print(algo.predict(uid = 'A39HTATAQ9V7YF', iid = '0205616461'))

    # - recommendation user case 2: in the customer and product preference database, what are the list of products to
    #                                recommend to the user 
    top_n = get_generic_top_n(predictions, n=10)
    
    # Print the recommended items for each user  - example for first 20 observations
    print("> Results:")
    for uid, user_ratings in top_n[0:20].items():
        print(uid, [iid for (iid, _) in user_ratings])
