# EXP15: FASHION-MNIST COMPARISON BETWEEN LEARNING_RATES, BATCH_SIZE, EPOCHS

from sklearn.datasets import fetch_openml
data = fetch_openml('Fashion-MNIST', version = 1, as_frame = False)
X = data.data
y = data.target.astype(int)

print(X.shape[0])
print(X.shape[1])
print(data.feature_names)

X = X / 255.0

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 3000, test_size = 500, random_state = 42, stratify = y)

result = []
learning_rates = [0.001, 0.01]
batch_size = [32, 64]
epochs = [200, 300]

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
for lr in learning_rates:
  for bs in batch_size:
    for ep in epochs:
      model = MLPClassifier(hidden_layer_sizes = (100,), learning_rate_init = lr, batch_size = bs, max_iter = ep, random_state = 42)
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      acc = accuracy_score(y_test, y_pred)
      result.append([lr, bs, ep, acc])

print("Learning Rates | Epochs | Batch Size | Accuracy")
for r in result:
  print(r[0], "       ", r[1], "       ", r[2], "       ", r[3])

def printres(x):
  return x[3]

best = max(result, key = printres)
print("The Best Hyperparameters: ")
print("Learning Rates: ", best[0])
print("Batch Size: ", best[1])
print("Epochs: ", best[2])
print("Accuracy: ", best[3])
