import pandas as pd 
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


file_csv = pd.read_csv('McDonalds-Yelp-Sentiment-DFE.csv', encoding='cp1252')
columns = ['policies_violated','review']
file_csv = file_csv[columns]
file_csv = file_csv[pd.notnull(file_csv['policies_violated'])]
file_csv.columns = ['policies_violated', 'review']

categories = []
for policie in file_csv.policies_violated:
	categories.append(policie.split())

categories = MultiLabelBinarizer().fit_transform(categories)



'''file_csv['category_id'] = file_csv['policies_violated'].factorize()[0]
category_id = file_csv[['policies_violated', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id.values)
id_to_categoru = dict(category_id[['category_id', 'policies_violated']].values)


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(file_csv.policies_violated).toarray()
labels = file_csv.category_id'''


X_train, X_test, y_train, y_test = train_test_split(file_csv.review, categories)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = OneVsRestClassifier(LinearSVC())
'''clf.fit(X_train_tfidf, y_train)

X_test_counts = count_vect.transform(X_test)
y_pred = clf.predict(X_test_counts)


print(clf.score(X_test_counts, y_test))'''

text_clf = Pipeline([('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', clf)])


parameters = {'vect__ngram_range': [(1, 1)],
             'tfidf__use_idf': (True, False),
             'vect__stop_words': ['english'],
             'clf__estimator__class_weight': ['balanced'],
             'clf__estimator__C': [1.0],
             'clf__estimator__multi_class': ['ovr', 'crammer_singer']
               }

print(text_clf.get_params().keys())


gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf.fit(X_train, y_train)

print(gs_clf.best_score_)
print(gs_clf.best_params_)





'''
data_train = file_csv[['review']]
categories = file_csv[['policies_violated']]


text = []
for comment in data_train.values:
	text.append(*comment)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(text)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test = X_train_tfidf


text = []
for categorie in categories.values:
	text.append(*categorie)

for i in range(0,len(text)-1):
	text[i] = str(text[i]).split()

text = MultiLabelBinarizer().fit_transform(text)


clf = OneVsRestClassifier(SVC(kernel='linear', class_weight='balanced'))
clf.fit(X_train_tfidf[:10], text[:10])
clf.fit(X_train_tfidf[:50], text[:50])
clf.fit(X_train_tfidf[:100], text[:100])
clf.fit(X_train_tfidf[:200], text[:200])
clf.fit(X_train_tfidf[:500], text[:500])
clf.fit(X_train_tfidf[:1000], text[:1000])


result = clf.predict(X_test)
#print(result)
#print(text[1250])
print(clf.score(X_test, result))'''
