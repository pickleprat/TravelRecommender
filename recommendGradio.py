import pandas as pd 
import numpy as np 

import os
import gradio as gr 
import pickle 
import math 

from fuzzywuzzy import fuzz 
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
        self.cities = [match.lower().strip() for match in list(self.city_data.city.unique())]
        model_pkl.close()

    def pmatch(self, city, threshold = 0.85):
        best_score = 0 
        best_match = None
        for match in self.cities: 
            score = fuzz.ratio(city, match)
            if score > best_score: 
                best_score = score
                best_match = match 

        best_match = best_match if best_score > threshold else city 
        

    def distance(self, lat1, lon1, lat2, lon2):
        R = 6371 
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        return distance

    
    def recommend(self, text):
        ud = []
        df_copy = self.df.copy()
        text_iter = simple_preprocess(text)
        vector = np.zeros((1, 180))
        city = None 
        for word in text_iter: 

            if self.city_map.get(word, None): city = word 
            try: vector += self.model.wv[word] 
            except Exception as E: ud.append(word)
        
        df_copy['similarity_score'] = df_copy.iloc[:, 1:181].apply(lambda x: cosine_similarity(x.values.reshape(1, -1), vector.reshape(1, -1))[0][0], axis = 1)

        if city is not None:
            if city not in self.cities: raise CityNotFoundException()
            longitude, latitude = self.city_data.loc[self.city_data.city.str.lower() == city, ['longitude', 'latitude']].values[0]
            
            df_copy['distance'] = df_copy.apply(lambda x: self.distance(x.latitude, x.longitude, latitude, longitude), axis = 1)
            df_copy = df_copy.sort_values(by = 'distance', ascending = True).iloc[:50, :]
            df_copy = df_copy.sort_values(by = 'similarity_score', ascending=False).iloc[:20, :]
            
            
            return df_copy.loc[:, ['place', 'city', 'pos', 'similarity_score', 'distance']]
        
        else :
            df_copy = df_copy.sort_values(by = 'similarity_score', ascending=False).iloc[:20, :]
            return df_copy.loc[:, ['place', 'city', 'pos', 'similarity_score']]

recommender = Recommender()

def recommend(text: str):
    try:
        result = recommender.recommend(text)
        return result.to_html()
    except Exception as e:
        return str(e)

inputs = gr.inputs.Textbox(label="Enter your text here")
outputs = gr.outputs.HTML()

title = "City Recommender"
description = "This app recommends places in India based on the description you provide"


app = gr.Interface(recommend, inputs, outputs, title=title, description=description)
app.launch(share = True)