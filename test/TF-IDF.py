import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


vector_corpus = [
    "يعيش محمد في جمهورية مصر العربية",
    "يعيش أحمد في المملكة العربية السعودية",
    "يتقابلان مرة كل عام في مصر"
]

arabicStopwords = stopwords.words("arabic")

# print(arabicStopwords)

vectorizer = TfidfVectorizer(stop_words=arabicStopwords)

tfidf = vectorizer.fit_transform(vector_corpus)

print(vectorizer.get_feature_names_out())

print(tfidf.shape)

import pandas as pd
print(pd.DataFrame(tfidf.toarray()))

