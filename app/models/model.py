#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Загрузка библиотек


# In[2]:


import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


# In[3]:


# Загрузка данных


# In[4]:


df = pd.read_csv('./heart_failure_clinical_records_dataset.csv')


# In[5]:


# Создание списков переменных


# In[6]:


features = list(df.columns)[:-1]
target = df.columns[-1]


# In[7]:


# Разделение на тренировочную и тестовую выборки, сохранение выборок


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=17)

X_test.to_csv("X_test.csv", index=None)
y_test.to_csv("y_test.csv", index=None)

X_train.to_csv("X_train.csv", index=None)
y_train.to_csv("y_train.csv", index=None)


# In[9]:


# Сборка пайплайна


# In[10]:


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


continuous_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
                   'serum_creatinine', 'serum_sodium', 'time']
base_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

continuous_transformers = []
base_transformers = []

for cont_col in continuous_cols:
    transformer = Pipeline([
        ('selector', NumberSelector(key=cont_col)),
        ('standard', StandardScaler())])
    continuous_transformers.append((cont_col, transformer))

for base_col in base_cols:
    base_transformer = Pipeline([
        ('selector', NumberSelector(key=base_col))])
    base_transformers.append((base_col, base_transformer))


# In[11]:


# Объединение трансформеров


# In[12]:


feats = FeatureUnion(continuous_transformers + base_transformers)
feature_processing = Pipeline([('feats', feats)])

feature_processing.fit_transform(X_train)


# In[13]:


# Классификатор


# In[14]:


pipeline = Pipeline([
    ('features', feats),
    ('classifier', RandomForestClassifier(random_state=17))])


# In[15]:


# Обучение пайплайна


# In[16]:


pipeline.fit(X_train, y_train)


# In[17]:


# Сохранение пайплайна


# In[18]:


with open("classifier_pipeline.dill", "wb") as f:
    dill.dump(pipeline, f)

