import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Read the file
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = None, names={"sepal_length_in_cm", "sepal_width_in_cm","petal_length_in_cm","petal_width_in_cm", "class"})

#Encoding the categorical column
dataset = dataset.replace({"class":  {"Iris-setosa":1,"Iris-versicolor":2, "Iris-virginica":3}})

#Checking the Correlation Between the columns
plt.figure(1)
sns.heatmap(dataset.corr())
plt.title('Correlation On iris Classes')
#----------------------------------------------
plt.figure(2)
plt.subplot(2, 2, 1)
plt.scatter(dataset["sepal_length_in_cm"], dataset["class"])
plt.title("Scatter PLot Sepal length and class")
plt.xlabel('Sepal Length')
plt.ylabel("Class")
plt.subplot(2, 2, 2)
plt.scatter(dataset["sepal_width_in_cm"], dataset["class"])
plt.title("Scatter PLot Sepal Width and class")
plt.xlabel('Sepal Width')
plt.ylabel("Class")
plt.subplot(2, 2, 3)
plt.scatter(dataset["petal_length_in_cm"], dataset["class"])
plt.title("Scatter PLot Petal length and class")
plt.xlabel('Petal Length')
plt.ylabel("Class")
plt.subplot(2, 2, 4)
plt.scatter(dataset["petal_width_in_cm"], dataset["class"])
plt.title("Scatter PLot Petal Width and class")
plt.xlabel('Petal Width')
plt.ylabel("Class")
plt.show()
#----------------------------------

X = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1].values

#-----------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Making a SVC
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

for i in range(len(y_test)):
        if y_test[i]!= y_pred[i]:
            print("\n Este valor es incorrecto en la posicion:",i) 
            print(y_test[i],y_pred[i])
            print("\n")
        else:
            print(y_test[i],y_pred[i])
        
plt.figure(3)
plt.plot(y_test,'bo', color= 'blue',label='True Values')
plt.plot(y_pred,'bo', color='red', label='Predicted Values')
plt.legend()
plt.show()