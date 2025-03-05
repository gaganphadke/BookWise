import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# 📥 Load datasets
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Ratings.csv')
users = pd.read_csv('Users.csv')

# 🛠 Fill NaN values to prevent errors
books.fillna('', inplace=True)
ratings.fillna(0, inplace=True)
users.fillna('', inplace=True)

# 🏷️ Combine features for TF-IDF
books['combined_features'] = books['Book-Title'] + ' ' + books['Book-Author'] + ' ' + books['Publisher']

# 🏷️ Create TF-IDF model
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined_features'])
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_model.pkl', 'wb'))

# 🔖 Create Book Indices for TF-IDF
book_indices = pd.Series(books.index, index=books['Book-Title']).to_dict()
pickle.dump(book_indices, open('book_indices.pkl', 'wb'))

# 🏷️ Merge datasets
ratings_with_title = ratings.merge(books[['Book-Title', 'ISBN']], on='ISBN')

# 🏷️ Filter books with sufficient ratings
book_rating_counts = ratings_with_title['Book-Title'].value_counts()
popular_books = book_rating_counts[book_rating_counts >= 50].index
filtered_ratings = ratings_with_title[ratings_with_title['Book-Title'].isin(popular_books)]

# 🛠 Create Pivot Table
pt = filtered_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating', fill_value=0)
pickle.dump(pt, open('pt.pkl', 'wb'))

# 🏷️ Create Popular Books DataFrame
popular_books_df = filtered_ratings.groupby('Book-Title').agg({
    'Book-Rating': 'mean',
    'User-ID': 'count'
}).rename(columns={'User-ID': 'Rating-Count'}).reset_index()
popular_books_df = popular_books_df.merge(books[['Book-Title', 'Book-Author', 'Image-URL-M']], on='Book-Title')
popular_books_df = popular_books_df.sort_values(by=['Rating-Count', 'Book-Rating'], ascending=False).head(50)
pickle.dump(popular_books_df, open('popular.pkl', 'wb'))

# 🔖 Save User Data
user_df = users[['User-ID', 'Location', 'Age']]
pickle.dump(user_df, open('user.pkl', 'wb'))

print("All .pkl files generated successfully! 🚀")
