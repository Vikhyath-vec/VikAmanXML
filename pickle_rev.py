import pickle
import pandas as pd
import random

# reviews = pickle.load(open('new_filtered_reviews.pickle', 'rb'))
# for review in reviews:
#     print(review['user'])

reviews = pd.read_pickle("Yelp/new_filtered_reviews.pickle")
print(len(reviews))
# for i, review in reviews.head(10).iterrows():
#     print(review['item'])

# reviews = pickle.load(open('Yelp/filtered_reviews.pickle', 'rb'))
# for i, review in reviews.head(10).iterrows():
#     print(review['item'])

# reviews = pickle.load(open('Yelp/reviews.pickle', 'rb'))
# print(reviews[0])
# for review in reviews:
#     print(review['user'])
    
# indices = range(len(reviews))
# train_indices = random.sample(indices, int(len(reviews)*0.8))
# valid_indices = random.sample(list(set(indices) - set(train_indices)), int(len(reviews)*0.1))
# test_indices = list(set(indices) - set(train_indices) - set(valid_indices))
# print(len(train_indices), len(valid_indices), len(test_indices), len(train_indices)+len(valid_indices)+len(test_indices))
# print(set(train_indices) & set(valid_indices))
# print(set(train_indices) & set(test_indices))
# print(set(valid_indices) & set(test_indices))

# with open('Yelp/6/train.index', 'w') as f:
#     f.write(' '.join([str(x) for x in train_indices]))
# with open('Yelp/6/validation.index', 'w') as f:
#     f.write(' '.join([str(x) for x in valid_indices]))
# with open('Yelp/6/test.index', 'w') as f:
#     f.write(' '.join([str(x) for x in test_indices]))

