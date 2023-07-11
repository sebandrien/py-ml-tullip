```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# Load the iris dataset
iris = load_iris()

# Create a DataFrame from the iris dataset for easier manipulation
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Split the dataset into a training set and a testing set
X_train, X_test, Y_train, Y_test = train_test_split(iris_df[iris['feature_names']], iris_df['target'], random_state=0)

# Normalize data and create a logistic regression model pipeline
pipeline = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', multi_class='auto'))

# Define the parameter grid for logistic regression
param_grid = {
    'logisticregression__C': np.logspace(-4, 4, 50),
    'logisticregression__penalty': ['l1', 'l2']
}

# Create GridSearchCV object
grid = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=0)

# Train the model
grid.fit(X_train, Y_train)

# Predict the test set results
Y_pred = grid.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)

# Cross validation score
cross_val_scores = cross_val_score(grid, X_train, Y_train, cv=5)

print("Model Best Parameters: ", grid.best_params_)
print(f"Model Accuracy: {accuracy}")
print("Cross validation scores: ", cross_val_scores)
print("Mean cross validation score: ", cross_val_scores.mean())
print("Classification Report: \n", classification_report(Y_test, Y_pred, target_names=iris.target_names))
```
