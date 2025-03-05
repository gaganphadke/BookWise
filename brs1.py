import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# 🟢 Load data with specified dtypes to fix DtypeWarning
books = pd.read_csv('books.csv', encoding='latin-1', dtype={
    'ISBN': str,
    'Book-Title': str,
    'Book-Author': str,
    'Year-Of-Publication': str,  # Load as string first to clean
    'Publisher': str,
    'Image-URL-S': str,
    'Image-URL-M': str,
    'Image-URL-L': str
})
ratings = pd.read_csv('ratings.csv', encoding='latin-1', dtype={
    'User-ID': int,
    'ISBN': str,
    'Book-Rating': int
})
users = pd.read_csv('users.csv', encoding='latin-1', dtype={
    'User-ID': int,
    'Location': str,
    'Age': float  # Use float to handle missing values
})

# 🟢 Clean Year-Of-Publication column
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
books.fillna('', inplace=True)

# 🟢 Merge ratings with books
merged_data = pd.merge(ratings, books, on='ISBN')
merged_data.fillna(0, inplace=True)

# 🟢 Filter to reduce size and fix PerformanceWarning
# Only include users who rated at least 20 books and books with at least 50 ratings
filtered_users = merged_data['User-ID'].value_counts()[merged_data['User-ID'].value_counts() >= 20].index
filtered_books = merged_data['Book-Title'].value_counts()[merged_data['Book-Title'].value_counts() >= 50].index
filtered_data = merged_data[merged_data['User-ID'].isin(filtered_users) & merged_data['Book-Title'].isin(filtered_books)]

# 🟢 Create a user-item matrix with a sparse format to save memory
user_item_matrix = filtered_data.pivot_table(index='User-ID', columns='Book-Title', values='Book-Rating', fill_value=0)
sparse_matrix = csr_matrix(user_item_matrix)


# 🟢 Print all unique user IDs in the new matrix
print("All User IDs in the filtered user-item matrix:")
print(user_item_matrix.index.tolist())


# 🟢 KNN model for collaborative filtering
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
model_knn.fit(sparse_matrix)

# 🟢 SVD for matrix factorization with optimized components
svd = TruncatedSVD(n_components=20)
svd_matrix = svd.fit_transform(sparse_matrix)

# 🟢 KNN Recommendations
def get_knn_recommendations(user_id, num_recommendations=5):
    if user_id not in user_item_matrix.index:
        return ["User ID not found."]
    user_index = user_item_matrix.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(sparse_matrix[user_index], n_neighbors=num_recommendations + 1)
   # 🟢 Fix for IndexError: Use valid column indices only
    recommendations = [
    user_item_matrix.columns[i] for i in indices.flatten() if i < len(user_item_matrix.columns)
][1:]

    return recommendations

# 🟢 SVD Recommendations
def get_svd_recommendations(user_id, num_recommendations=5):
    if user_id not in user_item_matrix.index:
        return ["User ID not found."]
    user_index = user_item_matrix.index.get_loc(user_id)
    user_vector = svd_matrix[user_index].reshape(1, -1)
    scores = svd_matrix.dot(user_vector.T).flatten()
    top_indices = scores.argsort()[-num_recommendations - 1:][::-1]
    recommendations = [
    user_item_matrix.columns[i] for i in top_indices 
    if i != user_index and i < len(user_item_matrix.columns)
]

    return recommendations

# 🟢 Test Recommendations
print("KNN Recommendations for User 95932:")
print(get_knn_recommendations(95932))

print("\nSVD Recommendations for User 95932:")
print(get_svd_recommendations(95932))


# 🟢 Optimized Content-Based Recommendations
books['Content'] = (
    books['Book-Title'].fillna('') + ' ' +
    books['Book-Author'].fillna('') + ' ' +
    books['Publisher'].fillna('') + ' ' +
    books['Year-Of-Publication'].fillna('').astype(str)
)

tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(books['Content']).astype('float32')

@lru_cache(maxsize=100)
def get_content_based_recommendations(title, num_recommendations=5):
    if title not in books['Book-Title'].values:
        return ["Book title not found."]
    idx = books.loc[books['Book-Title'] == title].index[0]
    sim_scores = tfidf_matrix.dot(tfidf_matrix[idx].T).toarray().flatten()
    book_indices = sim_scores.argsort()[-num_recommendations - 1:][::-1][1:]
    return books['Book-Title'].iloc[book_indices].tolist()

# 🟢 Efficient User Profile
def get_user_profile(user_id):
    if user_id not in ratings['User-ID'].values:
        return {"Error": "User ID not found."}
    user_ratings = merged_data.loc[merged_data['User-ID'] == user_id]
    profile = {
        'Books Read': len(user_ratings),
        'Average Rating Given': round(user_ratings['Book-Rating'].mean(), 2),
        'Top Genres': user_ratings['Book-Title'].value_counts().head(5).index.tolist()
    }
    return profile

# 🟢 Optimized Feedback System
user_feedback = {}

def give_feedback(user_id, book_title, rating):
    if user_id not in user_item_matrix.index or book_title not in user_item_matrix.columns:
        return "Invalid user ID or book title."
    user_feedback[(user_id, book_title)] = rating
    user_item_matrix.loc[user_id, book_title] = rating
    return "Feedback recorded!"

# Collaborative Recommendations
print("KNN Recommendations for User 95932:")
print(get_knn_recommendations(95932))

print("\nSVD Recommendations for User 95932:")
print(get_svd_recommendations(95932))

# Content-Based Recommendations
print("\nContent-Based Recommendations for 'The Da Vinci Code':")
print(get_content_based_recommendations('The Da Vinci Code'))

# User Profile
print("\nUser Profile for 95932:")
print(get_user_profile(95932))

# Feedback Example
print("\nFeedback for 'The Da Vinci Code':")
print(give_feedback(95932, 'The Da Vinci Code', 5))


# 🟢 Plot rating distribution
import matplotlib.pyplot as plt
plt.hist(merged_data['Book-Rating'], bins=10, color='skyblue', edgecolor='black')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--')
plt.show()


# 🟢 Most read authors
most_read_authors = merged_data['Book-Author'].value_counts().head(5)
most_read_authors.plot(kind='bar', title='Most Read Authors', color='orange')
plt.xlabel('Author')
plt.ylabel('Read Count')
plt.grid(axis='y', linestyle='--')
plt.show()

# 🟢 Most rated books
top_rated_books = merged_data.groupby('Book-Title')['Book-Rating'].mean().sort_values(ascending=False).head(5)
top_rated_books.plot(kind='bar', title='Top Rated Books', color='green')
plt.xlabel('Book Title')
plt.ylabel('Average Rating')
plt.grid(axis='y', linestyle='--')
plt.show()

user_id = 243  # Example User ID
print(f"\n🔹 Similar books based on user preferences (KNN) for User {user_id}:")
knn_recs = get_knn_recommendations(user_id)
for i, book in enumerate(knn_recs, 1):
    print(f"{i}. {book}")

# 🟢 Print SVD-based Recommendations
print(f"\n🔹 Similar books based on user preferences (SVD) for User {user_id}:")
svd_recs = get_svd_recommendations(user_id)
for i, book in enumerate(svd_recs, 1):
    print(f"{i}. {book}")

# 🟢 Print Content-Based Recommendations using TF-IDF
book_title = 'The Da Vinci Code'  # Example Book Title
print(f"\n🔹 Books with similar content to '{book_title}' (Content-Based using TF-IDF):")
content_recs = get_content_based_recommendations(book_title)
for i, book in enumerate(content_recs, 1):
    print(f"{i}. {book}")
