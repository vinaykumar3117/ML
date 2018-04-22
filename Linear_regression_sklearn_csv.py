from sklearn import linear_model
import pandas as pd

data = pd.read_csv("california_housing_train.csv")

X = data[["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income"]]
y = data["median_house_value"]

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

predictions = lm.predict(X)
print predictions[0:5]
print lm.score(X,y)
print lm.coef_
print lm.intercept_
