import streamlit as st
import brs1 as ns  # Replace with your .py script converted from .ipynb

st.title("📚 Book Recommendation System")

# 🟢 User Inputs
user_id = st.text_input("Enter User ID:", value="")
book_title = st.text_input("Enter a Book Title:", value="")

# 🟢 Display Recommendations
if st.button("Get Recommendations"):
    # 🟢 KNN Recommendations
    st.write(f"### 🔹 Similar books based on user preferences (KNN) for User {user_id}:")
    knn_recs = ns.get_knn_recommendations(int(user_id))
    for i, book in enumerate(knn_recs, 1):
        st.write(f"{i}. {book}")
    
    # 🟢 SVD Recommendations
    st.write(f"### 🔹 Similar books based on user preferences (SVD) for User {user_id}:")
    svd_recs = ns.get_svd_recommendations(int(user_id))
    for i, book in enumerate(svd_recs, 1):
        st.write(f"{i}. {book}")
    
    # 🟢 Content-Based Recommendations
    st.write(f"### 🔹 Books with similar content to '{book_title}' (Content-Based using TF-IDF):")
    content_recs = ns.get_content_based_recommendations(book_title)
    for i, book in enumerate(content_recs, 1):
        st.write(f"{i}. {book}")



# streamlit run app1.py
