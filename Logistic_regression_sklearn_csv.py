from sklearn.linear_model import LogisticRegression
import pandas as pd

data = pd.read_csv("diabetes.csv")

X = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y = data["Outcome"]

logr = LogisticRegression()
model = logr.fit(X,y)

predictions = logr.predict(X)
print predictions[0:50]
print logr.score(X,y)