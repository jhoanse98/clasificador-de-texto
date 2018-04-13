import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer



file_csv = pd.read_csv('McDonalds-Yelp-Sentiment-DFE.csv', encoding='cp1252')
data_train = file_csv[['review']]
categories = file_csv[['policies_violated']]


text = []
for comment in data_train.values:
	text.append(*comment)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(text)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test = X_train_tfidf[1000:]


text = []
for categorie in categories.values:
	text.append(*categorie)

for i in range(0,len(text)-1):
	text[i] = str(text[i]).split()

text = MultiLabelBinarizer().fit_transform(text)


clf = OneVsRestClassifier(SVC(kernel='linear'))
clf.fit(X_train_tfidf[:1000], text[:1000])

result = clf.predict(X_test)
#print(result)
print(clf.score(X_train_tfidf[1000:], result))
