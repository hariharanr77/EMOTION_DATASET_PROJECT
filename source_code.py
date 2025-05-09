#!/usr/bin/env python
# coding: utf-8

# # **1.Setup**

# In[ ]:


get_ipython().system('pip install pandas numpy matplotlib seaborn wordcloud scikit-learn nltk snscrape')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
nltk.download('stopwords')
from nltk.corpus import stopwords


# # **2.Load Dataset**

# In[ ]:


from google.colab import drive
import pandas as pd

file_path = '/content/drive/MyDrive/NMDS/emotion_dataset.csv'
df = pd.read_csv(file_path)
print(df.head())


# # 3.Clean the **Text**

# In[ ]:


def clean_text(text):
  text = text.lower()
  text = re.sub(r'http\S+','',text)    #remove URLs
  text = re.sub(r'@\w+','',text)       #remove mentions
  text = re.sub(r'#\w+','',text)       #remove hashtags
  text = re.sub(r'[^a-z\s]', '', text)    # remove punctuation and numbers
  words = text.split()
  words = [word for word in words if word not in stopwords.words('english')]
  return ' '.join(words)

  df['clean_text'] = df['text'].apply(clean_text)
  df.head()




# # **4.Exploratory Data Analysis(EDA)**

# In[ ]:


# After loading your dataset
df = pd.read_csv(file_path)

# Clean the column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()

# Split the single column into two columns
df[['clean_text', 'emotion']] = df[df.columns[0]].str.split(';', expand=True)

# Check if splitting worked
print(df.head())

# Now you can plot without issues
plt.figure(figsize=(10,5))
sns.countplot(x='emotion', data=df)
plt.title('Emotion Distribution')
plt.xticks(rotation=45)
plt.show()

# Generate wordcloud
text = ' '.join(df['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.xticks(rotation=45)
plt.axis('off')
plt.show()


# # **5.Feature Engineering(TF-IDF)**

# In[ ]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # **6.Model Building**

# In[ ]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# # **7.Model Evaluation**

# In[ ]:


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification classification_report:\n",classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

