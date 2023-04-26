import pandas as pd 
import numpy as np 

import gradio
import os
import pickle 
import math 

from gensim.utils import simple_preprocess
from IPython.display import display 
from sklearn.metrics.pairwise import cosine_similarity 

class CityNotFoundException(Exception):

    def __init__(self, message = "City does not exist in the database"):
        super().__init__(self, message)

class Recommender:

    def __init__(self):
        model_pkl = open(os.path.join('dep', 'word_vectoriser.pkl'), 'rb') 
        self.df = pd.read_csv(os.path.join('dep', 'DATASET_WITH_CATEGORY.csv'))
        self.city_data = pd.read_csv(os.path.join('dep', 'city_data.csv'))
        self.model = pickle.load(model_pkl)
        self.city_map = {city.lower().strip(): True for city in self.city_data['city']}
        model_pkl.close()

    def distance(self, lat1, lon1, lat2, lon2):
        R = 6371 
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        return distance

    
    def recommend(self, text, city = None, vicinity = False):
        ud = []
        df_copy = self.df.copy()
        text_iter = simple_preprocess(text)
        vector = np.zeros((1, 180))

        for word in text_iter: 
            try: 
                vector += self.model.wv[word]
            except Exception as E:
                ud.append(word)
        
        df_copy['similarity_score'] = df_copy.iloc[:, 1:181].apply(lambda x: cosine_similarity(x.values.reshape(1, -1), vector.reshape(1, -1))[0][0], axis = 1)
        

        if vicinity and city is not None:
            if city not in list(self.city_data.city.unique()): raise CityNotFoundException()
            longitude, latitude = self.city_data.loc[self.city_data.city == city, ['longitude', 'latitude']].values[0]
            
            df_copy['distance'] = df_copy.apply(lambda x: self.distance(x.latitude, x.longitude, latitude, longitude), axis = 1)
            df_copy = df_copy.sort_values(by = 'distance', ascending = True).iloc[:200, :]
        
            df_copy = df_copy.sort_values(by = 'similarity_score', ascending=False).iloc[:20, :]
            
            return df_copy.loc[:, ['place', 'city', 'pos', 'similarity_score', 'distance']]
        
        elif not vicinity:

            df_copy = df_copy.sort_values(by = 'similarity_score', ascending=False).iloc[:20, :]

            return df_copy.loc[:, ['place', 'city', 'pos', 'similarity_score']]

