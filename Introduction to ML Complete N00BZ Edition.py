#!/usr/bin/env python
# coding: utf-8

# In[63]:


import matplotlib
matplotlib.use('Agg')

from datascience import Table
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
plt.style.use('fivethirtyeight')

from annoy import AnnoyIndex


# In[5]:


songs = Table.read_table('/home/jovyan/work/datasets/msd_genre_dataset.csv')
songs


# In[10]:


attributes = songs.drop('genre','title','artist_name','track_id','genre')
attributes


# In[13]:


songs.take(5)


# In[16]:


songs.take(50000)


# In[20]:


songs.group('genre')


# In[24]:


songs.where('genre','metal')


# In[112]:


songs.where('genre','metal').group('artist_name').show(100)


# In[52]:


def distance(pt1, pt2):
    return np.sqrt(sum((pt1 - pt2) ** 2))
    
def row_distance(row1,row2):
    return distance( np.array(row1),  np.array(row2))

def majority_class(neighbors):
    return neighbors.group('genre').sort('count', descending=True).column('genre').item(0)

def classify(training, example, k):
    nearest_neighbors = closest(training, example, k)
    return majority_class(nearest_neighbors)

def distances(training,example):
    dists = []
    attributes = training.drop('genre','title','artist_name','track_id','genre')
    for row in attributes.rows:
        dist = row_distance(row, example)
        dists.append(dist)
    return training.with_column('distance',dists)

def closest(training, example, k):
    return distances(training, example).sort('genre').take(np.arange(k))


# In[110]:


metal_song_1 = songs.where('genre','metal').drop('genre','title','artist_name','track_id','genre').row(1)
metal_song_2 = songs.where('genre','metal').drop('genre','title','artist_name','track_id','genre').row(2)
pop_song_1 = songs.where('genre','pop').drop('genre','title','artist_name','track_id','genre').row(1)
pop_song_2 = songs.where('genre','pop').drop('genre','title','artist_name','track_id','genre').row(2)


# In[36]:


row_distance(metal_song_1,metal_song_2)


# In[38]:


row_distance(pop_song_1,pop_song_2)


# In[47]:


row_distance(metal_song_1,pop_song_2)


# In[49]:


songs.take(9)


# In[55]:


example_song = songs.drop('genre','title','artist_name','track_id','genre').row(9)
example_song

closest(songs.exclude(9),example_song,5)


# In[56]:


classify(songs.exclude(9), example_song, 5)


# In[46]:


songs.num_rows


# In[68]:


def evaluate_accuracy(training, test, k):
    test_attributes = songs.drop('genre','title','artist_name','track_id','genre')
    numcorrect = 0
    for i in range(test.num_rows):
        test_song = test_attributes.row(i)
        c = classify(training, test_song, k)
        if c == test.column('genre').item(i):
            numcorrect = numcorrect + 1
    return numcorrect / test.num_rows

shuffled = songs.sample(with_replacement=False)


# In[ ]:


trainset = shuffled.take(range(0,29800))
testset = shuffled.take(range(29800, 59600))
evaluate_accuracy(trainset,testset,1)


# In[69]:


trainset_tiny = shuffled.take(range(0,500))
testset_tiny = shuffled.take(range(500, 1000))
evaluate_accuracy(trainset_tiny,testset_tiny,5)


# In[100]:


f = 30
t = AnnoyIndex(f)  # Length of item vector that will be indexed
for i in range(5000):
    song = songs.drop('genre','title','artist_name','track_id','genre').take(i).to_array()[0]
    t.add_item(i, song)

t.build(10) # 10 trees
# t.save('test.ann')

# u = AnnoyIndex(f)
# u.load('test.ann') # super fast, will just mmap the file
print(t.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors


# In[107]:


print(songs.take(0).select('genre', 'artist_name', 'title'))
print(songs.take(65).select('genre', 'artist_name', 'title'))
print(songs.take(723).select('genre', 'artist_name', 'title'))


# In[109]:


songs.where('artist_name','ghost')

