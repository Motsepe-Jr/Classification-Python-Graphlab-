#!/usr/bin/env python
# coding: utf-8

# ## Analysing product sentiment
# 

# In[ ]:


import graphlab


# In[3]:


# load data 


# In[ ]:


products = graphlab.SFrame('amazon_baby.gl/')


# In[ ]:


products.head()


# In[ ]:


# creating 


# ## Build the word count vector for each review

# In[ ]:


products['word_count'] = graphlab.text_analytics.count_words(products['review']) # you can also do binary_words count
# creating a new column with number of counts for review column


# In[ ]:


products.head()


# In[4]:


# Exploring th most popular products


# In[ ]:


graphlab.canvas.set_target('ipynb')


# In[ ]:


products['name'].show()


# ## Explore popular product (Vulli Sophie)

# In[ ]:


giraffe_reviews = products[products['name'] == 'Vuli Sophie the Giraffe Teather']


# In[ ]:


len(giraffe_reviews)


# In[ ]:


giraffe_reviews['rating'].show(view='categorical')


# In[ ]:


# building a sentiment classifier 


# In[ ]:


# what is positive and negative sentiment?
# Ingnore all the three star reviews


# In[ ]:


# products = products[~products['rating'] == 3]
products = products[products['rating'] != 3]


# In[ ]:


# positive sentiment == 4 or 5 reviews


# In[ ]:


# positive = products[(products['rating']) == 4 & (products['rating'])]


# In[ ]:


products['sentiment'] = products['rating'] >= 4


# In[ ]:


# Lets train out sentiment classifer


# In[ ]:


train_data, test_data = products.random.split(.8, seed=0)


# In[ ]:


sentiment_model = graphlan.logistic_classifier.create(train_data, target ='sentiment', 
                                                     features=['word_count'],
                                                     validation_set= test_data)


# In[ ]:


# lets evaluate the sentiment model


# In[ ]:


sentiment_model.evaluate(test_data, matric='roc_curve')


# In[ ]:


sentiment_model.show(view='Evaluation')


# In[ ]:


# applying model to find most positive and negative reviews(sentiment) for a productss


# In[ ]:


giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[ ]:


## sort reviews based on predicted sentiment and explore


# In[ ]:


giraffe_reviews =giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[ ]:


giraffe_sentiment.head()


# In[ ]:


giraffe_reviews[0]['review']


# In[ ]:


giraffe_reviews[1]['review']


# In[ ]:


## show most negative reviews from the griraffe review


# In[ ]:


giraffe_review[-1]['review']


# In[ ]:


giraffe_review[-2]['review']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




