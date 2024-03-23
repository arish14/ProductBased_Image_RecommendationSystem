import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from tensorflow import keras
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Dropout, Flatten, Dense
#from keras import applications
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
from sklearn.metrics.pairwise import cosine_similarity

ssl._create_default_https_context = ssl._create_unverified_context
certifi.where()

fashion_df = pd.read_csv("/Users/arishbhayani/Desktop/Capstone/data/New_DF.csv")

boys_extracted_features = np.load('/Users/arishbhayani/Desktop/Capstone/Website/Model Features/Boys_ResNet_features.npy')
boys_Productids = np.load('/Users/arishbhayani/Desktop/Capstone/Website/Model Features/Boys_ResNet_feature_product_ids.npy')
girls_extracted_features = np.load('/Users/arishbhayani/Desktop/Capstone/Website/Model Features/Girls_ResNet_features.npy')
girls_Productids = np.load('/Users/arishbhayani/Desktop/Capstone/Website/Model Features/Girls_ResNet_feature_product_ids.npy')
men_extracted_features = np.load('/Users/arishbhayani/Desktop/Capstone/Website/Model Features/Men_ResNet_features.npy')
men_Productids = np.load('/Users/arishbhayani/Desktop/Capstone/Website/Model Features/Men_ResNet_feature_product_ids.npy')
women_extracted_features = np.load('/Users/arishbhayani/Desktop/Capstone/Website/Model Features/Women_ResNet_features.npy')
women_Productids = np.load('/Users/arishbhayani/Desktop/Capstone/Website/Model Features/Women_ResNet_feature_product_ids.npy')
fashion_df["ProductId"] = fashion_df["ProductId"].astype(str)

def get_similar_products_cnn(product_id, num_results=5):
    # Ensure product_id exists in the dataset
    if product_id not in fashion_df['ProductId'].values:
        raise ValueError(f"Product ID {product_id} not found in the dataset.")

    # Initialize Productids
    Productids = []

    # Determine the gender and set extracted features accordingly
    gender = fashion_df[fashion_df['ProductId'] == product_id]['Gender'].values[0]
    if gender == "Boys":
        extracted_features = boys_extracted_features
        Productids = boys_Productids
    elif gender == "Girls":
        extracted_features = girls_extracted_features
        Productids = girls_Productids
    elif gender == "Men":
        extracted_features = men_extracted_features
        Productids = men_Productids
    elif gender == "Women":
        extracted_features = women_extracted_features
        Productids = women_Productids

    # Default value in case no condition matches
    if len(Productids) == 0:
        Productids = []

    # Find the index of the input product
    doc_id = np.where(Productids == product_id)[0][0]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(extracted_features, extracted_features[doc_id].reshape(1, -1))

    # Get indices of the most similar products
    indices = np.argsort(similarity_scores.flatten())[::-1][:num_results]

    # Remove the index of the input product from the indices
    input_product_index = np.where(Productids == product_id)[0][0]
    indices = np.delete(indices, np.where(indices == input_product_index))

    # Input product details
    ip_row = fashion_df[['ImageURL', 'ProductTitle', 'price', 'rating']].loc[
        fashion_df['ProductId'] == Productids[indices[0]]]

    # Similar products sorted by 'rating' and filter out products with a rating of 0
    input_product_id = Productids[indices[0]]
    sim_rows = fashion_df[['ImageURL', 'ProductTitle', 'price', 'rating']].loc[
        (fashion_df['ProductId'].isin(list(np.array(Productids)[indices]))) &
        (fashion_df['rating'] > 0) &
        (fashion_df['ProductId'] != input_product_id)]

    # Get similarity scores for similar products
    similarity_scores = similarity_scores.flatten()[indices]

    return ip_row, sim_rows, similarity_scores
