import numpy as np
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize as wt
from scipy import sparse
import pickle as pkl
import os.path



def load_corpus(filename):
   
    with open(filename,"r") as f:
        lines = f.readlines()
        
    #stemmer = SnowballStemmer("french", ignore_stopwords=True)        
    
    X = []
    y = []
    for line in lines:
        y.append(line.split()[0][-2])
        #X.append(' '.join([stemmer.stem(m) for m in line.split()[1:]]))
        X.append(' '.join(line.split()[1:]))
    resultat = (np.array(X),np.array(y))
    
    return resultat




def construction_dico(X,Xt):
    #token = r"\b[^\d\W]+\b/g"
    #stemmer = SnowballStemmer("french", ignore_stopwords=False)
    vectorizer = CountVectorizer(stop_words=stopwords.words('french'),token_pattern=r'[a-zA-Z]+')
    Xvec = vectorizer.fit_transform(X)
    XvecT = vectorizer.transform(Xt)

    return Xvec,XvecT,vectorizer.get_feature_names()
    
    

def random_predict(Xt):
    file = open("predicte_random.txt","w")
    for i in Xt:
        prediction = np.random.choice(["C","M"])
        file.write(prediction+"\n")
    file.close()
    
def write_prediction(prediction):
    file = open("predicte_svm.txt","w")
    for p in prediction:
        file.write(p+"\n")
    file.close()

def lissage_prediction(prediction):
    
    for i in range(5,len(prediction)-5):
        if(sum([ 1 for p in prediction[i-5:i+4] if p == 'C'])/10)>0.65:
            prediction[i] = "M"
        else:
            prediction[i] = "C"
    
    for i in range(1,len(prediction)-1):
        if prediction[i]!=prediction[i-1] and prediction[i]!=prediction[i+1]:
            prediction[i] = "C" if prediction[i] == "M" else "M"
    
    return prediction

if __name__ == "__main__":
    
    X,y = load_corpus("data/president/corpus.tache1.learn.utf8")
    Xt,yt = load_corpus("data/president/corpus.tache1.test.utf8")
    (Xvec,XvecT,vect) = construction_dico(X,Xt)
    # données ultra basiques, à remplacer par vos corpus vectorisés

    #random_predict(Xt)
    # SVM
    clf = svm.LinearSVC()
    # Naive Bayes
    #clf = nb.MultinomialNB()x
    # regression logistique
    #clf = lin.LogisticRegression()
    
    # apprentissage
    clf.fit(Xvec, y)  
    #print(clf.predict([[2., 2.]])) # usage sur une nouvelle donnée
    predict =clf.predict(XvecT)
    predict=lissage_prediction(predict)
    write_prediction(predict)