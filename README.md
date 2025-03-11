# 📚 BookWise  

BookWise is a **book recommendation system** that helps users discover books based on popularity and personalized recommendations. Built using **Flask, Pandas, and Scikit-Learn**, the system provides an intuitive interface for exploring books.

## 🚀 Features  

- 🔥 **Top 50 Books**: Displays the most popular books based on votes and ratings.  
- 🎯 **Personalized Recommendations**: Get book suggestions based on a given book title.  
- 🔍 **Search Functionality**: Users can enter a book name to find related recommendations.  
- 🌐 **Web-Based UI**: A clean and user-friendly interface built with HTML, CSS, and Flask.  

## 🛠 Tech Stack  

- **Backend**: Flask, Python  
- **Frontend**: HTML, CSS, Bootstrap  
- **Data Processing**: Pandas, Scikit-Learn  
- **Database**: Pickle files (popular.pkl, pt.pkl, books.pkl, etc.)  

## ⚡ Installation & Setup  

### 1️⃣ Clone the repository  

git clone https://github.com/gaganphadke/BookWise.git
cd BookWise


### 2️⃣ Create a virtual environment (Optional but recommended)  

python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows


### 3️⃣ Install dependencies  

pip install -r requirements.txt


### 4️⃣ Run the Flask app  
python app.py

- The app will run at **http://127.0.0.1:5000/**  

## 🖥 Demo  

### 1️⃣ Homepage  
- Displays the top 50 books based on popularity.  

### 2️⃣ Search & Recommend  
- Enter a book title in the search bar to get recommendations.  

### 3️⃣ Book Recommendations  
- The system suggests books based on collaborative filtering.  

