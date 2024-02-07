import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

X_train = pd.read_csv('corpus.txt')
y_train = pd.read_csv('labels.txt')
X_test = pd.read_csv('test.txt')
y_test = pd.read_csv('test_labels.txt')

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Assuming you have your data and labels (X, y) ready
train_data = train_data.dropna()
test_data = test_data.dropna()

X_train = train_data['text']
y_train = train_data['label']

X_test = test_data['text']
y_test = test_data['label']

# Create TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform on training data, and transform test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Define the SVC model
svc = SVC()

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_test_tfidf, y_test)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Test the performance of the best model on the test set 
y_pred = best_model.predict(X_train_tfidf)

# Evaluate the performance
accuracy = accuracy_score(y_train, y_pred)
report = classification_report(y_train, y_pred)

# Print the results
print("Best Parameters:", best_params)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", report)
