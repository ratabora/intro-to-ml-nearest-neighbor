#!/usr/bin/env python
# coding: utf-8

# In[31]:


import matplotlib
matplotlib.use('Agg')

from datascience import Table
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
plt.style.use('fivethirtyeight')


# In[36]:


banknotes = Table.read_table('/home/jovyan/work/datasets/data_banknote_authentication.csv')
banknotes


# In[17]:


banknotes.scatter('variance','skewness',colors='class')


# In[18]:


banknotes.scatter('skewness','entropy',colors='class')


# In[35]:


fig = plt.figure(figsize=(8,8))
Axes3D(fig).scatter(
    banknotes.column('skewness'),
    banknotes.column('variance'),
    banknotes.column('curtosis'), 
    c=banknotes.column('class'),
    cmap='viridis',
    s=50
)


# In[ ]:




