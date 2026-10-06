from sklearn.datasets import fetch_openml
ld = fetch_openml('Fashion-MNIST', version = 1, as_frame = False)
X = ld.data
y = ld.target.astype(int)
print(X.shape[0])
print(X.shape[1])
print(ld.feature_names)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 3000, test_size = 300, random_state = 42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC
models = {"Linear":SVC(kernel = 'linear'), "Polynomial":SVC(kernel = 'poly', degree = 3), "RBF":SVC(kernel = 'rbf')}

results = {}
from sklearn.metrics import accuracy_score
print("\n")
for name, model in models.items():
  print("Training",name,"SVM...")
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  results[name] = acc
print("\n")
print("Linear SVM: ",results["Linear"])
print("Polynomial SVM: ",results["Polynomial"])
print("RBF SVM: ",results["RBF"])
print("\n")
print("COMPARISON")
if results["Linear"] > results["Polynomial"] and results["RBF"]:
  print("Linear SVM has highest accuracy")
elif results["Polynomial"] > results["Linear"] and results["RBF"]:
  print("Polynomial SVM has highest accuracy")
else:
  print("RBF SVM has highest accuracy")
