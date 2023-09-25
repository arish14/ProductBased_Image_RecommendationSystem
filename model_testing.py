import numpy as np
import pandas as pd
from PIL import Image
import urllib.request
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity, manhattan_distances


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

def get_similar_products(product_id, num_results, similarity_metric='euclidean'):
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
    Productids = list(Productids)
    doc_id = Productids.index(product_id)
    
    if similarity_metric == 'cosine':
        pairwise_dist = cosine_similarity(extracted_features, extracted_features[doc_id].reshape(1, -1))
    elif similarity_metric == 'manhattan':
        pairwise_dist = manhattan_distances(extracted_features, extracted_features[doc_id].reshape(1, -1))
    elif similarity_metric == 'correlation':
        pairwise_dist = pairwise_distances(extracted_features, extracted_features[doc_id].reshape(1, -1), metric='correlation')
    else:
        # Default to Euclidean distance if the metric is not recognized
        pairwise_dist = pairwise_distances(extracted_features, extracted_features[doc_id].reshape(1, -1))
    
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists = np.sort(pairwise_dist.flatten())[0:num_results]

    print("=" * 20, "Input item details", "=" * 20)
    ip_row = fashion_df[['ImageURL', 'ProductTitle']].loc[fashion_df['ProductId'] == Productids[indices[0]]]
    for indx, row in ip_row.iterrows():
        image = Image.open(urllib.request.urlopen(row['ImageURL']))
        image = image.resize((224, 224))
        image.show()
        print(f"Product Title: {row['ProductTitle']}")

    print("\n", "=" * 20, f"Top {num_results} Recommended items", "=" * 20)
    for i in range(1, len(indices)):
        rows = fashion_df[['ImageURL', 'ProductTitle']].loc[fashion_df['ProductId'] == Productids[indices[i]]]
        for indx, row in rows.iterrows():
            image = Image.open(urllib.request.urlopen(row['ImageURL']))
            image = image.resize((224, 224))
            image.show()
            print(f"Product Title: {row['ProductTitle']}")
            print(f"Similarity using {similarity_metric.capitalize()}: {1 - pdists[i]}")

print("# Visual Similarity based Recommendation")

user_input1 = input("Enter the item id: ")
user_input2 = int(input("Enter number of products to be recommended: "))
user_input3 = input("Enter similarity metric (cosine, manhattan, correlation, or euclidean): ")

get_similar_products(user_input1, user_input2, user_input3.lower())
