import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load your DataFrame
df = pd.read_csv("/Users/arishbhayani/Desktop/Capstone/data/New_DF.csv")

# Define the rating scale
reader = Reader(rating_scale=(0, 5))


data = Dataset.load_from_df(df[['user_id', 'ProductId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Create and train the SVD model (you can choose other algorithms as well)
model = SVD()
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')



# User Input
user_input1 = input("Enter the item id: ")
user_input2 = int(input("Enter number of products to be recommended: "))

# Get a list of all unique product IDs in your DataFrame
unique_product_ids = df['ProductId'].unique()

# Create a list to store predictions for the user
user_predictions = []

# Iterate over all unique product IDs and get predictions for the user
for product_id in unique_product_ids:
    prediction = model.predict(user_input1, product_id)
    user_predictions.append((product_id, prediction.est))

# Sort the predictions by estimated rating in descending order
user_predictions.sort(key=lambda x: x[1], reverse=True)

# Print the top N recommended products
N = user_input2  # Use the user's input for the number of recommendations
top_recommendations = user_predictions[:N]

print(f"Top {N} Recommendations for User {user_input1}:")
for product_id, estimated_rating in top_recommendations:
    print(f"Product ID: {product_id}, Estimated Rating: {estimated_rating}")
