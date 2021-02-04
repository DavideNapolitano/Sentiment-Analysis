import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords as sw
from nltk.stem.snowball import ItalianStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm

df=pd.read_csv('development.csv')
df_e=pd.read_csv('evaluation.csv')

print('\nDevelopment\n')
# print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())
X=df.text.values
y=df.clas.values
length=df.text.index

print('\nEvaluation\n')
# print(df_e.head())
print(df_e.info())
print(df_e.describe())
print(df_e.isna().sum())
X_e=df_e.text.values


stop_w=sw.words('italian')
stemmer=ItalianStemmer()
to_rep=['>','@','<','|','&','#','%','"','*',':',';','=','[',']','{','}','(',')',',','.','+','-',
        '_','\'']
to_ex=['a','b','c','d','e','f','g','h','i','l','m','n','o','p','q','r','s','t','u','v','z','ciò','oltre','e',
       'o','inoltre','ma','però','dunque','anzi','che','davvero','spesso','ieri','oggi','domani','mai',
       'notte','giorno','mattina','pomeriggio','sera','avra', 'avro', 'cio', 'fara', 'faro', 'perche', 'pero', 'piu',
       'sara', 'saro', 'stara', 'staro','ok']
num=['0','1','2','3','4','5','6','7','8','9']
stop_w.extend(to_ex)
en_char=['w','y','j','x','k']
vocali=['aa','ee','ii','oo','uu']
voc=['a','e','i','o','u']
stop_w.extend(en_char)
to_add=['non','più','si','ma','ne','contro','però','ok']
for ta in to_add:
    stop_w.remove(ta)

# -----------------------------------------------------Analysis---------------------------------------------------------

print(df['clas'].value_counts())
sns.countplot(x='clas',data=df)
plt.show()
color = [0 if lab=='pos' else 1 for lab in y]
vect = TfidfVectorizer( min_df=2,ngram_range=(1,2))
X_plot = vect.fit_transform(X)
X_plot = TruncatedSVD(n_components=2).fit_transform(X_plot)
plt.scatter(X_plot[:,0],X_plot[:,1], c=color)
plt.title("Raw Data")
plt.show()

# --------------------------------------------------Preprocessing-------------------------------------------------------

print("\nPreprocessing\n")

def clean(text):
    text_list=[]
    for t in word_tokenize(text):
        t = t.strip()
        for i in to_rep:
            if t.find(i) != -1:
                t = t.replace(i, ' ')
        text_list.append(str(t.lower()))
    list = []
    for el in text_list:
        for number in num:
            if el.find(number) != -1:
                el = ''.join([i for i in el if not i.isdigit()])
        spl=el.split()
        for sub_spl in spl:
            list.append(sub_spl)
    cutlen2=[]
    for parola in list:
        if len(parola)>2 and len(parola)<16:
            if not any((c in en_char) for c in parola):
                c = 0
                for el in vocali:
                    if parola.find(el) != -1:
                        c = c + 1
                if c==0:
                    if parola.find("issim")!= -1:
                        x=parola.split("issim")
                        cutlen2.append(x[0])
                    elif parola.find("mente")!= -1:
                        x = parola.split("mente")
                        tmp1 = x[0]
                        last_char = tmp1[-1:]
                        if last_char not in voc:
                            tmp1 = tmp1 + 'e'
                        cutlen2.append(tmp1)
                    else:
                        cutlen2.append(parola)
        if parola=='ok':
            cutlen2.append(parola)
        if parola == "!":
            if parola not in cutlen2:
                cutlen2.append(parola)
        if parola == "?":
            if parola not in cutlen2:
                cutlen2.append(parola)
    clean_s = ' '.join(cutlen2)
    # clean_mess = [stemmer.stem(word) for word in clean_s.split() if word.lower() not in stop_w]
    clean_mess = [stemmer.stem(word) for word in clean_s.split()]
    # clean_mess = [word for word in clean_s.split() if word.lower() not in stop_w] # Senza Stemming
    return clean_mess

list_rev=[]
for i in range(len(X)):
    lemma=[]
    test = clean(X[i])
    LtS=' '.join([str(elem) for elem in test])
    list_rev.append(LtS)

list_rev_e=[]
for i in range(len(X_e)):
    lemma=[]
    test = clean(X_e[i])
    LtS=' '.join([str(elem) for elem in test])
    list_rev_e.append(LtS)

color = [0 if lab=='pos' else 1 for lab in y]
vect = TfidfVectorizer(min_df=2,ngram_range=(1,2))
X_plot = vect.fit_transform(list_rev)
X_plot = TruncatedSVD(n_components=2).fit_transform(X_plot)
plt.scatter(X_plot[:,0],X_plot[:,1], c=color)
plt.title("Cleaned Data")
plt.show()

# %%
print("\nClassification\n")

X1_train, X1_test, y1_train, y1_test=train_test_split(list_rev, y, test_size=0.2, random_state=42)

#GRIDSEARCHCV
print("     GridSearchCV")
pipe = Pipeline(steps=[ ('tf', TfidfVectorizer(   use_idf=True,
                                                  min_df=2,
                                                  ngram_range=(1,2)
                                                  )),
                        ('kb', SelectKBest(chi2)),
                        ('svc', svm.LinearSVC(dual=False)),
                       ])
param_grid = {
    'kb__k': np.arange(34000,36001,500),
    'svc__C': np.arange(0.7,1.51,0.1)
}
search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, cv=5, scoring="f1_weighted", verbose=3)
search.fit(X1_train, y1_train)
print(f"best_score: {search.best_score_}")
print(f"best_par: {search.best_params_}")
best=search.best_estimator_

clf1=best.fit(X1_train,y1_train)
y_pred1=clf1.predict(X1_test)

f=f1_score(y1_test, y_pred1, average='weighted')  # VALUTAZIONE
cm=confusion_matrix(y1_test,y_pred1)
print(cm)
print(f)

# -----------------------------------------------------Evaluation-------------------------------------------------------
# %%
print("\nEvaluation\n")

clf_e=best.fit(list_rev,y)
y_pred_EVAL=clf_e.predict(list_rev_e)

color = [0 if lab=='pos' else 1 for lab in y_pred_EVAL]
vect = TfidfVectorizer(min_df=2,ngram_range=(1,2))
X_plot = vect.fit_transform(list_rev_e)
X_plot = TruncatedSVD(n_components=2).fit_transform(X_plot)
plt.scatter(X_plot[:,0],X_plot[:,1], c=color)
plt.title("Prediction")
plt.show()

with open("sol.csv","w",newline='') as csvfile:
    filewriter=csv.writer(csvfile)
    filewriter.writerow(['Id','Predicted'])
    for i in range(len(y_pred_EVAL)):
        line = [f'{i}', f'{y_pred_EVAL[i]}']
        filewriter.writerow(line)

print("Fine")