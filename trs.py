### Import necessary libraries

from typing import Dict, Text
import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs

import os
import pprint
import tempfile
import pandas as pd
import matplotlib.pyplot as plt

masterdf = pd.read_csv('Reviews.csv')
### standardize item data types, especially string, float, and integer

masterdf[['UserId',      
          'ProfileName',  
         ]] = masterdf[['UserId','ProfileName']].astype(str)

# we will play around with the data type of the quantity, 
# which you shall see later it affects the accuracy of the prediction.

masterdf['Score'] = masterdf['Score'].astype(float)
interactions_dict = masterdf.groupby(['UserId', 'ProfileName'])[ 'Score'].sum().reset_index()

## we tansform the table inta a dictionary , which then we feed into tensor slices
# this step is crucial as this will be the type of data fed into the embedding layers
interactions_dict = {name: np.array(value) for name, value in interactions_dict.items()}
interactions = tf.data.Dataset.from_tensor_slices(interactions_dict)

## we do similar step for item, where this is the reference table for items to be recommended
items_dict = masterdf[['ProfileName']].drop_duplicates()
items_dict = {name: np.array(value) for name, value in items_dict.items()}
items = tf.data.Dataset.from_tensor_slices(items_dict)

## map the features in interactions and items to an identifier that we will use throught the embedding layers
## do it for all the items in interaction and item table
## you may often get itemtype error, so that is why here i am casting the quantity type as float to ensure consistency
interactions = interactions.map(lambda x: {
    'UserId' : x['UserId'], 
    'ProfileName' : x['ProfileName'], 
    'Score' : float(x['Score']),
})

items = items.map(lambda x: x['ProfileName'])


unique_item_titles = np.unique(np.concatenate(list(items.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["UserId"]))))
### get unique item and user id's as a lookup table
unique_item_titles = np.unique(np.concatenate(list(items.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["UserId"]))))

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = interactions.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(60_000)
test = shuffled.skip(60_000).take(20_000)

class RetailModel(tfrs.Model):

    def __init__(self, user_model, item_model):
        super().__init__()
        ### Candidate model (item)
        ### This is Keras preprocessing layers to first convert user ids to integers, 
        ### and then convert those to user embeddings via an Embedding layer. 
        ### We use the list of unique user ids we computed earlier as a vocabulary:
        item_model = tf.keras.Sequential([
                                        tf.keras.layers.experimental.preprocessing.StringLookup(
                                        vocabulary=unique_item_titles, mask_token=None),
                                        tf.keras.layers.Embedding(len(unique_item_titles) + 1, embedding_dimension)
                                        ])
        ### we pass the embedding layer into item model
        self.item_model: tf.keras.Model = item_model
            
        ### Query model (users)    
        user_model = tf.keras.Sequential([
                                        tf.keras.layers.experimental.preprocessing.StringLookup(
                                        vocabulary=unique_user_ids, mask_token=None),
                                        # We add an additional embedding to account for unknown tokens.
                                        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
                                        ])
        self.user_model: tf.keras.Model = user_model
        
        ### for retrieval model. we take top-k accuracy as metrics
        metrics = tfrs.metrics.FactorizedTopK(candidates=items.batch(128).map(item_model))
        
      
        task = tfrs.tasks.Retrieval(
                                    metrics=metrics
                                    )
       
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["UserId"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.item_model(features["ProfileName"])
        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)

### Fitting and evaluating

### we choose the dimensionality of the query and candicate representation.
embedding_dimension = 32

## we pass the model, which is the same model we created in the query and candidate tower, into the model
item_model = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.StringLookup(
                                vocabulary=unique_item_titles, mask_token=None),
                                tf.keras.layers.Embedding(len(unique_item_titles) + 1, embedding_dimension)
                                ])

user_model = tf.keras.Sequential([
                                tf.keras.layers.experimental.preprocessing.StringLookup(
                                vocabulary=unique_user_ids, mask_token=None),
                                # We add an additional embedding to account for unknown tokens.
                                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
                                ])

model = RetailModel(user_model, item_model)

# a smaller learning rate may make the model move slower and prone to overfitting, so we stick to 0.1
# other optimizers, such as SGD and Adam, are listed here https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

## fit the model with ten epochs
model_hist = model.fit(cached_train, epochs=3)

recommends = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
recommends.index_from_dataset(tf.data.Dataset.zip((items.batch(100),items.batch(100).map(model.item_model))))


def predict(c):
    _, titles = recommends(tf.constant([c])) 
    print(f"Recommendations for user : {titles[0][:3]}") 
    print(type(titles))
    return(f"Recommendations for user : {titles[0][:3]}") 
    

if __name__=='__main__':
    #c = input("enter:")
    predict() 

#A1SP2KVKFXXRU1 [b'David C. Sullivan' b'Heather L. Koranteng "HK"' b'A. Hughes']
#A3IV7CL2C13K2U [b'Greg' b'LK' b'Elon Smith "Elon"']
#AQCY5KRO7489S  [b'Pat' b'Garrett' b'Joyce']

#A2F4LZVGFLD1OB   [b'Michael' b'DaisyH' b'Kimdoll']
#A1J87LOAYSMHO9  [b'P. Parchman "Savvy Shopper"' b'Nikki' b'Erin']
