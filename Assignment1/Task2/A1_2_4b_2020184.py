# Importing Libraries

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Reading the data with headings attached to it
X_train = pd.read_csv('Assignment1\Task2\\new_corpus.txt')
y_train = pd.read_csv('Assignment1\Task2\\new_labels.txt')
X_test = pd.read_csv('Assignment1\Task2\\generated_samples\\addfilehere.txt')

# X_train, X_test, y_train, y_test = train_test_split(data2["text"], data2["label"], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

svc = SVC()
param_grid = {               # Parameter grid for GridSearchCV
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print("Best Parameters:", best_params)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", report)