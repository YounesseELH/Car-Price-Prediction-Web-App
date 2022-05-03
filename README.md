
<div align="center">
  <strong><h1>Car Price Prediction Web App</h1></strong>
</div>
    
## License
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

## First Step :  Built Machine Learning Model 
[ 🔗  Open in Colab](https://colab.research.google.com/github/YounesseELH/Car-Price-Prediction-Web-App/blob/main/Car_Price_Prediction.ipynb)

 ## Steps that I followed to build machine learning model : 
 - Import Dependencies : 
 ```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
```
 - Data Collection and Processing

 ```python
df = pd.read_csv('/content/car data.csv')
```
 - Convert strings to numbers for categorical data

 ```python
#convert Text to numbers because machines speak numbers 
df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
df.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
df.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
```
 - Split Data into Training data and Test data

 ```python
 X = df.drop(['Car_Name','Selling_Price'],axis=1)
Y = df['Selling_Price']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=2)

```
 - Model Training : using Linear Regression

 ```python
ln_model = LinearRegression()
ln_model.fit(X_train,Y_train)
```
 - Model Evaluation

 ```python
train_pred = ln_model.predict(X_train)
err_score = metrics.r2_score(Y_train,train_pred)
print(err_score)
```
 - Visualize actual price and predicted price

 ```python
 #using data train
plt.scatter(Y_train,train_pred)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.show()

#using test data
test_pred = ln_model.predict(X_test)
err_score = metrics.r2_score(Y_test,test_pred)
print(err_score)

#graph
plt.scatter(Y_test,test_pred)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.show()
```

 - Lasso regression

 ```python
lasso_model = Lasso()

#using data train
lasso_model.fit(X_train,Y_train)
train_pred = lasso_model.predict(X_train)
err_score = metrics.r2_score(Y_train,train_pred)

print(err_score)

plt.scatter(Y_train,train_pred)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.show()

#using data test
test_pred = lasso_model.predict(X_test)
err_score = metrics.r2_score(Y_test,test_pred)
print(err_score)

#graph
plt.scatter(Y_test,test_pred)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.show()
```
