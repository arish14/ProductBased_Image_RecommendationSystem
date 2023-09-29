from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import sys
from Model_code.model_deploy import get_similar_products_cnn
import re
#from addproducts import addproducts_bp

app = Flask(__name__)

#app.register_blueprint(addproducts_bp)

df = pd.read_csv('/Users/arishbhayani/Desktop/Capstone/data/New_DF.csv')

@app.route('/search-page')
def search_page():
    return render_template('search.html')



@app.route('/')
def home():
    #For home page initial design 
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('search', '')
        escaped_query = re.escape(query)
        pattern = r'\b' + escaped_query + r'\b'
        #results = df[df['ProductTitle'].str.contains(query, case=False)]
        results = df[df['ProductTitle'].str.contains(pattern, case=False, regex=True)]
        return render_template('search_results.html', results=results)
    print(request.form.get('product_id'), file=sys.stderr)
    return render_template('/search.html')
    


@app.route('/product_search', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        product_id = request.form.get('product_id')
        return redirect(url_for('display_similar_products', product_id=product_id))
    
    return render_template('product_search.html')

@app.route('/recommend', methods=['GET', 'POST'])
def display_similar_products():
    if request.method == 'POST':
        product_id = request.form.get('product_id')
        num_results = 5
        ip_row, sim_rows = get_similar_products_cnn(product_id, num_results)
    
        return render_template('similar_products.html', ip_row=ip_row, sim_rows=sim_rows)

    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run()




