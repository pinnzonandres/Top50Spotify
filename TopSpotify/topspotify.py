import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""### READ THE DATA"""

df=pd.read_csv('top50.csv',encoding="ISO-8859-1")

df.head()

data = df.iloc[:,1:]

"""### Searching Missing Data"""

col_names= data.columns.tolist()
for column in col_names:
  print("Valores Nulos en <{0}>: <{1}>".format(column,data[column].isnull().sum()))

"""#### Most popular Artist"""

artist = [artist for artist, df in data.groupby(['Artist.Name'])]
plt.figure(figsize=(15,8))
plt.bar(artist, data.groupby(['Artist.Name']).sum()['Popularity'])
plt.xticks(artist, rotation = 'vertical', size= 8 )
plt.grid()
plt.show

plt.figure(figsize=(15,8))
plt.bar(artist, data.groupby(['Artist.Name']).count()['Popularity'])
plt.xticks(artist, rotation = 'vertical', size= 8 )
plt.grid()
plt.show

"""#### Most Popular Genre"""

genre = [genre for genre, df in data.groupby(['Genre'])]
plt.figure(figsize=(15,8))
plt.bar(genre, data.groupby(['Genre']).sum()['Popularity'])
plt.xticks(genre, rotation = 'vertical', size= 8 )
plt.grid()
plt.show

"""#### Popularity By Beats per Minute"""

beats=[beat for beat, df in data.groupby(['Beats.Per.Minute'])]
plt.figure(figsize=(20,10))
plt.plot(beats, data.groupby(['Beats.Per.Minute']).sum()['Popularity'])
plt.xticks(beats, size= 8)
plt.xlabel('Beats Per Minute')
plt.ylabel('Sum of Popularity per Beat')
plt.grid()
plt.show()

plt.figure(figsize=(20,10))
plt.bar(beats, data.groupby(['Beats.Per.Minute']).count()['Popularity'])
plt.xticks(beats,rotation = 'vertical', size= 8)
plt.xlabel('Beats Per Minute')
plt.ylabel('Popularity per Beat')
plt.grid()
plt.show()

"""#### Relation between Numerical Data and Popularity"""

plt.figure(figsize=(20,20))
for i in range(3,len(col_names)-2):
    plt.subplot(4,2,i-2)
    plt.scatter(data[col_names[i]],data['Popularity'])
    plt.xlabel(col_names[i])
    plt.ylabel('Popularity')
plt.show()

"""#### Correlation"""

import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot= True)
plt.show()

data['Popularity'].describe()

"""#### Making a Gruouping Class by Popularity"""

def classing(a):
    list=np.zeros(len(a))
    for i in range(len(a)):
        list[i]= int((a[i]-(a.min()))/5)+1
        if list[i]==6:
            list[i]=5
    return list

data['PopularityClass']= classing(data['Popularity'])

data.head()

data['PopularityClass'].value_counts()

"""#### Encoding the Genre Data"""

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
data['GenreEncoder'] = labelencoder_X.fit_transform(data['Genre'])

data.head()

data.dtypes

num_Data = data.select_dtypes(exclude = 'object')

num_Data.head()

X_R = num_Data.drop(columns=['Popularity','PopularityClass'])
X_C = num_Data.drop(columns=['Popularity','PopularityClass'])
Y_R = num_Data['Popularity'].values
Y_C = num_Data['PopularityClass'].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_R= sc.fit_transform(X_R)

from sklearn.model_selection import train_test_split

X_R_train,X_R_test,Y_R_train,Y_R_test = train_test_split(X_R,Y_R, test_size= 0.2, random_state= 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_R_train, Y_R_train)

"""## Predicting the Test set results"""

Y_R_pred = regressor.predict(X_R_test)

"""## Evaluating the Model Performance"""

from sklearn.metrics import r2_score
print('The R2 SCORE IS')
print(r2_score(Y_R_test, Y_R_pred))

plt.plot(Y_R_test,color='red',label='Y_TEST')
plt.plot(Y_R_pred, color='green', label='Y_Pred')
plt.legend()
plt.show()

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_R_train, Y_R_train)

"""## Predicting the Test set results"""

Y_R_pred = regressor.predict(X_R_test)
print('The R2 SCORE IS')

print(r2_score(Y_R_test, Y_R_pred))
plt.plot(Y_R_test,color='red',label='Y_TEST')
plt.plot(Y_R_pred, color='green', label='Y_Pred')
plt.legend()
plt.show()

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_R_train, Y_R_train)

"""## Predicting the Test set results"""

Y_R_pred = (regressor.predict(sc.transform(X_R_test)))
print('The R2 SCORE IS')

print(r2_score(Y_R_test, Y_R_pred))
plt.plot(Y_R_test,color='red',label='Y_TEST')
plt.plot(Y_R_pred, color='green', label='Y_Pred')
plt.legend()
plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_R_train)
regressorpoly = LinearRegression()
regressorpoly.fit(X_poly, Y_R_train)

"""## Predicting the Test set results"""

Y_R_pred = regressorpoly.predict(poly_reg.transform(X_R_test))
print('The R2 SCORE IS')

print(r2_score(Y_R_test, Y_R_pred))
plt.plot(Y_R_test,color='red',label='Y_TEST')
plt.plot(Y_R_pred, color='green', label='Y_Pred')
plt.legend()
plt.show()

X_C_train,X_C_test,Y_C_train,Y_C_test = train_test_split(X_C,Y_C, test_size= 0.2)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma= 0.2)
classifier.fit(X_C_train, Y_C_train)
# Predicting the Test set results
y_pred = classifier.predict(X_C_test)

"""**Cross Validation**"""

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_C_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_C_train, y = Y_C_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

