from flask import Flask,render_template,url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import jolib
import pickle
import joblib

#loading model from the disk
filename="nlp_model.pkl"
clf=pickle.load(open(filename,"rb"))
cv=pickle.load(open("transform.pkl", "rb"))
app=Flask (__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    df=pd.read_csv("spam.csv",encoding="latin-1")
    df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)
    #Features and labels
    df["label"]=df["class"].map({"ham":0,"spam":1})
    x=df["message"]
    y=df["label"]
    
    #Extract Feature with countVectorizer
    cv=CountVectorizer()
    x=cv.fit_transform(x) #fit the data
    
    pickle.dump(cv,open("transform.pkl","wb"))
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=42)
    
    #NaiveBayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    
    clf=MultinomialNB()
    clf.fit(x_train,y_train)
    clf.score(x_test,y_test)
    filename="nlp_model.pkl"
    pickle.dump(clf,"NB_spam_model.pkl","rb")
    
    #Alternative usage of saved model
    joblib.dump(clf, "NB_spam_model.pkl")
    NB_spam_model=open("NB_spam_model.pkl","rb")
    clf=joblib.load(NB_spam_model)
    
    if request.method == 'POST':
        message =request.form["message"]
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction=clf.predict(vect)
        
    return render_template("result.html",prediciton=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)