import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
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
X_train, X_test, y_train, y_test = train_test_split(file_csv.review, categories)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = OneVsRestClassifier(LinearSVC(class_weight='balanced'))


text_clf = Pipeline([('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', clf)])


parameters = {'vect__ngram_range': [(1, 1)],
             'tfidf__use_idf': [True],
             'vect__stop_words': ['english'],
             'clf__estimator__class_weight': ['balanced'],
             'clf__estimator__C': [0.3],
             'clf__estimator__loss': ['hinge'],
             'vect__binary': [True],
             'vect__max_features': [18000],
             'vect__max_df': [1000]
               }

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf.fit(X_train, y_train)

print(gs_clf.best_score_)
print(gs_clf.best_params_)





