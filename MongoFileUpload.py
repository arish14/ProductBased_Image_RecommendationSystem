from pymongo import MongoClient
import certifi
import urllib 
import pandas as pd

# Replace with your MongoDB Atlas connection string
connection_string = "mongodb+srv://wine_review:" + urllib.parse.quote("NP9TU3sY64NbqSf1") + "@mongoproject.miuch.mongodb.net/"

csv_file_path = "/Users/arishbhayani/Desktop/Capstone/data/New_DF.csv"

# Name of the collection in MongoDB where you want to insert the data
collection_name = "Product"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Connect to MongoDB Atlas with tlsCAFile option
client = MongoClient(connection_string, tlsCAFile=certifi.where())

# Access your database
db = client["MongoProject"]

# Access the collection where you want to insert the data
collection = db[collection_name]

# Convert the DataFrame to a list of dictionaries (one dictionary per row)
data = df.to_dict(orient="records")

# Insert the data into the MongoDB collection
result = collection.insert_many(data)

# Print the inserted document IDs
print("Inserted document IDs:", result.inserted_ids)
