from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import sys
from Model_code.model_deploy import get_similar_products_cnn
import re
from pymongo import MongoClient
import certifi
import urllib
import pymongo

#from addproducts import addproducts_bp

app = Flask(__name__)

connection_string = "Connection String"


collection_name = "Product"

# Connect to MongoDB Atlas with tlsCAFile option
client = MongoClient(connection_string, tlsCAFile=certifi.where())

# Access your database
db = client["MongoProject"]

# Access the collection where you inserted the data
collection = db[collection_name]

#app.register_blueprint(addproducts_bp)

df = pd.read_csv('/Users/arishbhayani/Desktop/Capstone/data/New_DF.csv')

@app.route('/search-page')
def search_page():
    return render_template('search.html')



@app.route('/')
def home():
    #For home page initial design 
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('search', '')
        escaped_query = re.escape(query)
        pattern = r'\b' + escaped_query + r'\b'

        # Perform a search query on the MongoDB collection
        # Filter by "ProductTitle" matching the pattern and "rating" greater than 0
        results = collection.find(
            {"ProductTitle": {"$regex": pattern, "$options": "i"},
             "rating": {"$gt": 0}}
        ).sort("rating", -1).limit(20)  # Limit results to 20
        #print(request.form.get('product_id'), file=sys.stderr)
        return render_template('search_results.html', results=results)

    print(request.form.get('product_id'), file=sys.stderr)
    return render_template('/search.html')


    


@app.route('/product_search', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        product_id = request.form.get('product_id')
        return redirect(url_for('display_similar_products', product_id=product_id))
    
    return render_template('product_search.html')

@app.route('/recommend', methods=['POST'])
def display_similar_products():
    if request.method == 'POST':
        product_id = request.form.get('product_id')
        num_results = 5
        ip_row, sim_rows, similarity_scores = get_similar_products_cnn(product_id, num_results)

        # Fetch details of the selected product
        selected_product = collection.find_one({"ProductId": product_id})

        # Convert the DataFrame to a list of dictionaries for sim_rows
        sim_rows = sim_rows.to_dict('records')

        # Print similarity scores on the console
        print("Similarity Scores:", similarity_scores, file=sys.stderr)

        return render_template('similar_products.html', ip_row=selected_product, sim_rows=sim_rows)

    return redirect(url_for('index'))

 




if __name__ == '__main__':
    app.run()
