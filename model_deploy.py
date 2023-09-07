import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.metrics import pairwise_distances
import requests
from PIL import Image
import pickle
from datetime import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import urllib3
urllib3.disable_warnings()
import certifi
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
certifi.where()

fashion_df = pd.read_csv("/Users/arishbhayani/Desktop/Capstone/data/New_DF.csv")

boys_extracted_features = np.load('./Boys_ResNet_features.npy')
boys_Productids = np.load('./Boys_ResNet_feature_product_ids.npy')
girls_extracted_features = np.load('./Girls_ResNet_features.npy')
girls_Productids = np.load('./Girls_ResNet_feature_product_ids.npy')
men_extracted_features = np.load('./Men_ResNet_features.npy')
men_Productids = np.load('./Men_ResNet_feature_product_ids.npy')
women_extracted_features = np.load('./Women_ResNet_features.npy')
women_Productids = np.load('./Women_ResNet_feature_product_ids.npy')
fashion_df["ProductId"] = fashion_df["ProductId"].astype(str)

def get_similar_products_cnn(product_id, num_results):
    Productids = []
    if fashion_df[fashion_df['ProductId'] == product_id]['Gender'].values[0] == "Boys":
        extracted_features = boys_extracted_features
        Productids = boys_Productids
    elif fashion_df[fashion_df['ProductId'] == product_id]['Gender'].values[0] == "Girls":
        extracted_features = girls_extracted_features
        Productids = girls_Productids
    elif fashion_df[fashion_df['ProductId'] == product_id]['Gender'].values[0] == "Men":
        extracted_features = men_extracted_features
        Productids = men_Productids
    elif fashion_df[fashion_df['ProductId'] == product_id]['Gender'].values[0] == "Women":
        extracted_features = women_extracted_features
        Productids = women_Productids
    
    if len(Productids) == 0:
        # Default value in case no condition matches
        Productids = []
    
    doc_id = np.where(Productids == product_id)[0][0]
    pairwise_dist = pairwise_distances(extracted_features, extracted_features[doc_id].reshape(1, -1))
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #print("=" * 20, "input product details", "=" * 20)
    ip_row = fashion_df[['ImageURL', 'ProductTitle', 'price', 'rating']].loc[
        fashion_df['ProductId'] == Productids[indices[0]]]
    sim_rows = fashion_df[['ImageURL', 'ProductTitle', 'price', 'rating']].loc[
        fashion_df['ProductId'].isin(list(np.array(Productids)[indices]))]
    
    return ip_row, sim_rows
