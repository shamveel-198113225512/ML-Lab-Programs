'''
Implement a Linear Support Vector Machine (SVM) to classify the Iris dataset. Visualize
the decision boundary and discuss how the margin is determined.
Tasks:
● Load and preprocess the Iris dataset.
● Implement a Linear SVM for binary classification (e.g., classify Setosa vs. Non-
Setosa).
● Visualize the decision boundary and margin.
● Discuss the concept of the margin and how it influences classification.
'''

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
X = X[:,[2,3]]
y = np.where(y == 0,1,0)
print(X.shape[0])
print(X.shape[1])
print(iris.feature_names)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model = SVC(kernel = "linear", C = 3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print("Accuracy: ",acc)
print("No.of support vectors: ",len(model.support_vectors_))
plt.figure(figsize = (8,6))
X_min = X_train[:,0].min()-1
X_max = X_train[:,0].max()+1
y_min = X_train[:,0].min()-1
y_max = X_train[:,0].max()+1

xx, yy = np.meshgrid(np.linspace(X_min, X_max, 500), np.linspace(X_min, X_max, 500))
z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.scatter(X_train[y_train == 1,0], X_train[y_train == 1,1], label = "Setosa")
plt.scatter(X_train[y_train == 0,0], X_train[y_train == 0,1], label = "Non-Setosa")
plt.contour(xx, yy, z, levels=[0], linewidths = 2)
plt.contour(xx, yy, z, levels=[-1,1], linestyles = "--")
plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s = 100, facecolors = "none", edgecolors = "black", label = "Supprt Vectors")
plt.xlabel('petal length(scaled)')
plt.ylabel('petal width(scaled)')
plt.title('Linear svm decision boundary and margin')
plt.legend()
plt.show()
