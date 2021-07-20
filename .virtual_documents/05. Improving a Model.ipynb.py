import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.get_params()


# create a evaluate prediction function 
def evaluate_pred(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs y_pred labels
    on a classification model.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    
    metric_dict = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1 score": round(f1, 2)
    }
    
    print(f"Accuracy: {accuracy * 100:.2f}get_ipython().run_line_magic("")", "")
    print(f"Precision: {precision * 100:.2f}get_ipython().run_line_magic("")", "")
    print(f"Recall: {recall * 100:.2f}get_ipython().run_line_magic("")", "")
    print(f"F1 Score: {f1 * 100:.2f}get_ipython().run_line_magic("")", "")
    
    return metric_dict


# import the heart disease dataset
heart_disease = pd.read_csv('data/heart-disease.csv')
heart_disease.info()


# shuffle the samples and divide into features X and samples y
np.random.seed(42)
heart_disease_shuffled = heart_disease.sample(frac=1)

X = heart_disease_shuffled.drop('target', axis=1)
y = heart_disease_shuffled['target']

# split the data into train, validate and test samples
train_split = round(0.70 * len(heart_disease_shuffled))
valid_split = round(train_split + 0.15 * len(heart_disease_shuffled))

X_train, y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split]
X_test,  y_test  = X[valid_split:], y[valid_split:]

# fitting the data to the model
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)

y_preds = clf.predict(X_valid)
baseline_metrics = evaluate_pred(y_valid, y_preds)


# create a second classifier with adjusted n_estimators
np.random.seed(42)
clf_2 = RandomForestClassifier(n_estimators=100)
clf_2.fit(X_train, y_train)
y_preds = clf_2.predict(X_valid)
evaluate_pred(y_valid, y_preds);






