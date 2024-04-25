from flask import Flask, render_template,request
import pickle
import numpy as np

popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
score = pickle.load(open('score.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    # Round the ratings to two decimal places
    rounded_ratings = [round(rating, 2) for rating in popular_df['avg_ratings'].values]
    
    return render_template('index.html',
                           bookname=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           ratings=rounded_ratings
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    
    if user_input:
        index = np.where(pt.index == user_input)[0]
        
        if index:
            index = index[0]
            similar_items = sorted(list(enumerate(score[index])), key=lambda x: x[1], reverse=True)[1:6]
            
            data = []
            for i in similar_items:
                item = []
                temp = books[books['Book-Title'] == pt.index[i[0]]]
                item.extend(list(temp.drop_duplicates('Book-Title')['Book-Title'].values))
                item.extend(list(temp.drop_duplicates('Book-Title')['Book-Author'].values))
                item.extend(list(temp.drop_duplicates('Book-Title')['Image-URL-M'].values))
                data.append(item)
            
            return render_template('recommend.html', data=data)
        else:
            return "Book not found."  
    else:
        return "Please provide a book title."  

if __name__ == '__main__':
    app.run(debug=True)
