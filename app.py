import pandas as pd 
import numpy as np 

from flask import Flask, render_template, request 
from Recommender import Recommender 
 
app = Flask(__name__)

app.jinja_env.globals.update(zip=zip, len = len)

global recommender 
recommender = Recommender()

class RecommendedPlace:

    def __init__(self, df: pd.DataFrame):
        self.df = df 
        self.col_count = len(self.df)
        self.place_json = None 
        self.place_json = self.make_json()
        

    def make_json(self):
        if self.col_count == 4:
            self.place_json = {
                "type":1, 
                "place":self.df["place"], 
                "city":self.df["city"], 
                "compound":self.df["pos"], 
                "similarity_score":self.df["similarity_score"]
            }
        else:
            self.place_json = {
                "type":2, 
                "place":self.df["place"], 
                "city":self.df["city"], 
                "compound":self.df["pos"], 
                "similarity_score":self.df["similarity_score"], 
                "distance":self.df["distance"], 
            }

        return self.place_json 
    
    @property
    def place(self):
        return self.place_json["place"]
    
    @property 
    def problem(self):
        return self.place_json["type"]
    
    @property
    def city(self):
        return self.place_json["city"]
    
    @property
    def distance(self):
        return self.place_json["distance"]
    
    @property
    def similarity_score(self):
        return self.place_json["similarity_score"]
    
    @property
    def compound(self):
        return self.place_json["compound"]
    
    @property 
    def type(self):
        return self.place_json["type"]
    
    def __repr__(self): 
        return self.place_json.__str__()
    

@app.route('/', methods = ['POST', 'GET'])
def home(): 
    cities_db = pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\Projects\place-recommendation\travel_recommender\app\Flask_App\dep\city_data.csv')
    params = {"iterations":0}
    params["cities"] = list(cities_db["city"].unique()) 

    if request.method == 'POST':
        description = request.form.get('description')
        city = request.form.get('city')
        vicinity = True if request.form.get('vicinity') == "True" else False 
        df = recommender.recommend(description,city, vicinity)
        places = [RecommendedPlace(df.iloc[row, :]) for row in range(len(df))]   
        params["places"] = places 
    
    return render_template('index.html', params = params)

if __name__ == '__main__':
    app.run(debug=True)