
# coding: utf-8

# In[137]:


import pandas as pd
d = pd.read_csv("YouTube-Spam-Collection-v1/Youtube01-Psy.csv")


# In[138]:


d.tail()


# In[139]:


len(d.query('CLASS == 1'))


# In[140]:


len(d.query('CLASS == 0'))


# In[141]:


len(d)


# In[142]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


# In[143]:


dvec = vectorizer.fit_transform(d['CONTENT'])


# In[144]:


dvec


# In[145]:


analyze = vectorizer.build_analyzer()


# In[146]:


print(d['CONTENT'][349])
analyze(d['CONTENT'][349])


# In[147]:


vectorizer.get_feature_names()


# In[148]:


dshuf = d.sample(frac=1)


# In[149]:


d_train = dshuf[:300]
d_test = dshuf[300:]
d_train_att = vectorizer.fit_transform(d_train['CONTENT']) # fit bag-of-words on training set
d_test_att = vectorizer.transform(d_test['CONTENT']) # reuse on testing set
d_train_label = d_train['CLASS']
d_test_label = d_test['CLASS']


# In[150]:


d_train_att


# In[151]:


d_test_att


# In[152]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=80)


# In[153]:


clf.fit(d_train_att, d_train_label)


# In[154]:


clf.score(d_test_att, d_test_label)


# In[155]:


from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(d_test_att)
confusion_matrix(d_test_label, pred_labels)


# In[156]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[157]:


# load all datasets and combine them
d = pd.concat([pd.read_csv("YouTube-Spam-Collection-v1/Youtube01-Psy.csv"),
               pd.read_csv("YouTube-Spam-Collection-v1/Youtube02-KatyPerry.csv"),
               pd.read_csv("YouTube-Spam-Collection-v1/Youtube03-LMFAO.csv"),
               pd.read_csv("YouTube-Spam-Collection-v1/Youtube04-Eminem.csv"),
               pd.read_csv("YouTube-Spam-Collection-v1/Youtube05-Shakira.csv")])


# In[158]:


len(d)


# In[159]:


len(d.query('CLASS == 1'))


# In[160]:


len(d.query('CLASS == 0'))


# In[161]:


dshuf = d.sample(frac=1)
d_content = dshuf['CONTENT']
d_label = dshuf['CLASS']


# In[162]:


# set up a pipeline
from sklearn.pipeline import Pipeline, make_pipeline
pipeline = Pipeline([
    ('bag-of-words', CountVectorizer()),
    ('random forest', RandomForestClassifier()),
])
pipeline


# In[163]:


# or: pipeline = make_pipeline(CountVectorizer(), RandomForestClassifier())
make_pipeline(CountVectorizer(), RandomForestClassifier())


# In[164]:


pipeline.fit(d_content[:1500],d_label[:1500])


# In[165]:


pipeline.score(d_content[1500:], d_label[1500:])


# In[166]:


pipeline.predict(["what a neat video!"])


# In[167]:


pipeline.predict(["plz subscribe to my channel"])


# In[168]:


scores = cross_val_score(pipeline, d_content, d_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[181]:


# add tfidf
from sklearn.feature_extraction.text import TfidfTransformer
pipeline2 = make_pipeline(CountVectorizer(),
                          TfidfTransformer(norm=None),
                          RandomForestClassifier())


# In[182]:


scores = cross_val_score(pipeline2, d_content, d_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[183]:


pipeline2.steps


# In[184]:


# parameter search
parameters = {
    'countvectorizer__max_features': (None, 1000, 2000),
    'countvectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'countvectorizer__stop_words': ('english', None),
    'tfidftransformer__use_idf': (True, False), # effectively turn on/off tfidf
    'randomforestclassifier__n_estimators': (20, 50, 100)
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipeline2, parameters, n_jobs=-1, verbose=1)


# In[185]:


grid_search.fit(d_content, d_label)


# In[186]:


print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

