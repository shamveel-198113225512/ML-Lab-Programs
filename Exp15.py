'''Implement and perform hyperparameter tuning for a neural network on the Fashion 
MNIST dataset. Experiment with different learning rates, batch sizes, and epochs, and 
discuss the impact on model performance. 
Tasks: 
● Load and preprocess the Fashion MNIST dataset. 
● Experiment with different hyperparameters (learning rate, batch size, epochs). 
● Train and evaluate the network. 
● Discuss how hyperparameter choices affect model performance.
'''

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X = data.data
y = data.target.astype(int)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=3000, test_size=500,
    random_state=42, stratify=y
)

result = []
learning_rates = [0.001, 0.01]
batch_size = [32, 64]
epochs = [200, 300]

for lr in learning_rates:
    for bs in batch_size:
        for ep in epochs:
            model = MLPClassifier(
                hidden_layer_sizes=(100,),
                learning_rate_init=lr,
                batch_size=bs,
                max_iter=ep,
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            result.append([lr, bs, ep, acc])
print("Learning Rate\tBatch Size\tEpochs\tAccuracy")

for lr, bs, ep, acc in result:
    print(lr, bs, ep, round(acc, 3), sep="\t")

def printres(x):
    return x[3]

best = max(result, key=printres)
print("\nThe Best Hyperparameters:")
print("Learning Rate:", best[0])
print("Batch Size:", best[1])
print("Epochs:", best[2])
print("Accuracy:", round(best[3], 3))