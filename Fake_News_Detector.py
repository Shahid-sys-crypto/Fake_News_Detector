import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("https://raw.githubusercontent.com/lutzhamel/fake-news/refs/heads/master/data/fake_or_real_news.csv")

nltk.download('stopwords')
nltk.download('punkt')

def cleaned_text(text):
    text=text.lower()
    text=re.sub(r"[^a-zA-Z0-9\s]","",text)
    tokens=word_tokenize(text)
    stop_words=set(stopwords.words('english'))
    tokens=[word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["cleaned_text"]=df["text"].apply(cleaned_text)

vectorizer=TfidfVectorizer(max_features=5000)
x=vectorizer.fit_transform(df["cleaned_text"])
y=df["label"].map({"REAL":1,"FAKE":0})

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(f"accuracy:{accuracy_score(y_test,y_pred)*100:.2f}")
print(classification_report(y_test,y_pred))

def predict_news(news):
    clean=cleaned_text(news)
    vectorized_text=vectorizer.transform([clean])
    prediction=model.predict(vectorized_text)
    return "REAL" if prediction[0]==1 else "FAKE"

def main():
    while(True):
        news=input("Enter news:")
        print(predict_news(news))
        choice=input("do you wanna continue(yes/no)?")
        if choice.lower!="yes":
            break

if __name__=="__main__":
    main()


