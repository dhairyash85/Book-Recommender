import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

books = pd.read_csv("../Data/Books.csv")
ratings = pd.read_csv("../Data/Ratings.csv")
users = pd.read_csv("../Data/Users.csv")

ratings_with_name=ratings.merge(books, on="ISBN")
num_rating_df=ratings_with_name.groupby("Book-Title").count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':"num_ratings"}, inplace=True)

avg_rating_df=ratings_with_name.groupby("Book-Title").mean('Book-Rating').reset_index()
avg_rating_df.rename(columns={'Book-Rating':"avg_ratings"}, inplace=True)

# Popularity
populariy_df=num_rating_df.merge(avg_rating_df, on="Book-Title")
populariy_df=populariy_df[populariy_df['num_ratings']>=250].sort_values('avg_ratings', ascending=False)
populariy_df=populariy_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_ratings']]

# Collaborative
x=ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
experienced_users=x[x].index
filtered_ratings=ratings_with_name[ratings_with_name['User-ID'].isin(experienced_users)]

y=filtered_ratings.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index
final_ratings=filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)].drop_duplicates()
pt=final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)
print(pt)

similarity_score=cosine_similarity(pt)

def recommend(title):
    index=np.where(pt.index==title)
    similar_items=sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:11]
    return similar_items

pickle.dump(populariy_df, open("popular.pkl", "wb"))