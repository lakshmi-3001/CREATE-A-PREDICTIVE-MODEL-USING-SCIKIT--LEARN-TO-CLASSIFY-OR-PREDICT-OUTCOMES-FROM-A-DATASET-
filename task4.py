import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import string
from sklearn.ensemble import RandomForestClassifier
nltk.download('stopwords')

#1. import the email dataset
df = pd.read_csv('/content/spam_ham_dataset.csv')
df['text'] = df['text'].apply(lambda x: x.replace('r\n', ' '))
df.info()

#2. To Preprocess the Dataset
# Map labels to binary (spam=1, ham=0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

#3. To Split Dataset into Training and Testing Sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#4. for making all the emails in steamed version(remove punctuation)
ps = PorterStemmer()
corpus =[]

stopword_set = set(stopwords.words('english'))
for i in range(0, len(X)):
     text = X.iloc[i]
     text = text.lower()
     text = text.translate(str.maketrans('', '', string.punctuation)).split()
     text = [ps.stem(word) for word in text if not word in stopword_set]
     text = ' '. join(text)
     corpus.append(text)

#5.  to check for the original text
df.text.iloc[0]

#6. To check for the steamed version
corpus[0]

#7. To Vectorize Text Data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_jobs=-1)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#8. evaluation of accuracy
clf.score(X_test, y_test)

#9. to classify email as spam or ham
email_to_classify = df.text.values[12]

email_to_classify

email_text = email_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
email_text = [ps.stem(word) for word in email_text if not word in stopword_set]
email_text = ' '.join(email_text)

email_corpus = [email_text]

X_email = vectorizer.transform(email_corpus)

#10. check the 12TH email is spam or ham.
clf.predict(X_email)

df.label_num.iloc[12]
