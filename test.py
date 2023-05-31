# KNN
print('KNN:')
# Import the libraries
print('Import the libraries:')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Create the features and labels
print('Create the features and labels:')
X = df['title']
y = df['target']
# Split the data into training and testing sets
print('Split the data into training and testing sets:')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the TfidfVectorizer
print('Initialize the TfidfVectorizer:')
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform the training data
print('Fit and transform the training data:')
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# Transform the test set
print('Transform the test set:')
tfidf_test = tfidf_vectorizer.transform(X_test)
# Initialize the KNN classifier
print('Initialize the KNN classifier:')
knn = KNeighborsClassifier(n_neighbors=2)
# Fit the model
print('Fit the model:')
knn.fit(tfidf_train, y_train)
# Predict on the test set
print('Predict on the test set:')
y_pred = knn.predict(tfidf_test)
# Print the accuracy score
print('Print the accuracy score:')
print('Accuracy score: {}'.format(accuracy_score(y_test, y_pred)))
# Print the confusion matrix
print('Print the confusion matrix:')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
# Print the classification report
print('Print the classification report:')
print('Classification report:')
print(classification_report(y_test, y_pred))

print(' ------------------------------------ ')
print(' ------------------------------------ ')

# KNN with different k values
print('KNN with different k values:')
# Import the libraries
print('Import the libraries:')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Create the features and labels
print('Create the features and labels:')
X = df['title']
y = df['target']
# Split the data into training and testing sets
print('Split the data into training and testing sets:')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the TfidfVectorizer
print('Initialize the TfidfVectorizer:')
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform the training data
print('Fit and transform the training data:')
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# Transform the test set
print('Transform the test set:')
tfidf_test = tfidf_vectorizer.transform(X_test)
# Initialize the KNN classifier
print('Initialize the KNN classifier:')
knn = KNeighborsClassifier(n_neighbors=2)
# Fit the model
print('Fit the model:')
knn.fit(tfidf_train, y_train)
# Predict on the test set
print('Predict on the test set:')
y_pred = knn.predict(tfidf_test)
# Print the accuracy score
print('Print the accuracy score:')
print('Accuracy score: {}'.format(accuracy_score(y_test, y_pred)))
# Print the confusion matrix
print('Print the confusion matrix:')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
# Print the classification report
print('Print the classification report:')
print('Classification report:')
print(classification_report(y_test, y_pred))

print(' ------------------------------------ ')
print(' ------------------------------------ ')

# KNN with different k values (k=5) and (k=10) and (k=15) and (k=20)

print('KNN with different k values (k=5) and (k=10) and (k=15) and (k=20):')
# Import the libraries
print('Import the libraries:')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Create the features and labels
print('Create the features and labels:')
X = df['title']
y = df['target']
# Split the data into training and testing sets
print('Split the data into training and testing sets:')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the TfidfVectorizer
print('Initialize the TfidfVectorizer:')
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform the training data
print('Fit and transform the training data:')
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# Transform the test set
print('Transform the test set:')
tfidf_test = tfidf_vectorizer.transform(X_test)
# Initialize the KNN classifier
print('Initialize the KNN classifier:')
knn = KNeighborsClassifier(n_neighbors=5)
# Fit the model
print('Fit the model:')
knn.fit(tfidf_train, y_train)
# Predict on the test set
print('Predict on the test set:')
y_pred = knn.predict(tfidf_test)
# Print the accuracy score
print('Print the accuracy score:')
print('Accuracy score: {}'.format(accuracy_score(y_test, y_pred)))
# Print the confusion matrix
print('Print the confusion matrix:')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
# Print the classification report
print('Print the classification report:')
print('Classification report:')
print(classification_report(y_test, y_pred))

print(' ------------------------------------ ')
print(' ------------------------------------ ')

print('KNN with different k values (k=5) and (k=10) and (k=15) and (k=20):')
# Import the libraries
print('Import the libraries:')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Create the features and labels
print('Create the features and labels:')
X = df['title']
y = df['target']
# Split the data into training and testing sets
print('Split the data into training and testing sets:')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the TfidfVectorizer
print('Initialize the TfidfVectorizer:')
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform the training data
print('Fit and transform the training data:')
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# Transform the test set
print('Transform the test set:')
tfidf_test = tfidf_vectorizer.transform(X_test)
# Initialize the KNN classifier
print('Initialize the KNN classifier:')
knn = KNeighborsClassifier(n_neighbors=10)
# Fit the model
print('Fit the model:')
knn.fit(tfidf_train, y_train)
# Predict on the test set
print('Predict on the test set:')
y_pred = knn.predict(tfidf_test)
# Print the accuracy score
print('Print the accuracy score:')
print('Accuracy score: {}'.format(accuracy_score(y_test, y_pred)))
# Print the confusion matrix
print('Print the confusion matrix:')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
# Print the classification report
print('Print the classification report:')
print('Classification report:')
print(classification_report(y_test, y_pred))

print(' ------------------------------------ ')
print(' ------------------------------------ ')

# MANUAL KNN with different k values (k=5) and (k=10) and (k=15) and (k=2
#Create the features and labels manually
print('Create the features and labels manually:')
X = df['title']
y = df['target']
# Split the data into training and testing sets
print('Split the data into training and testing sets:')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# TDIDF MANUAL IMPLEMENTATION
print('TDIDF MANUAL IMPLEMENTATION:')
# Create the features and labels manually
print('Create the features and labels manually:')
X = df['title']
y = df['target']
# Split the data into training and testing sets
print('Split the data into training and testing sets:')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create the vocabulary
print('Create the vocabulary:')
corpus = X_train
vocabulary = {}
for doc in corpus:
    for word in doc.split():
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
# Create the tfidf matrix
print('Create the tfidf matrix:')
tfidf_train = np.zeros((len(corpus), len(vocabulary)))
for i, doc in enumerate(corpus):
    for word in doc.split():
        tfidf_train[i, vocabulary[word]] += 1
# Create the tfidf matrix
print('Create the tfidf matrix:')
tfidf_test = np.zeros((len(X_test), len(vocabulary)))
for i, doc in enumerate(X_test):
    for word in doc.split():
        tfidf_test[i, vocabulary[word]] += 1
# KNN MANUAL IMPLEMENTATION
print('KNN MANUAL IMPLEMENTATION:')
# Create the features and labels manually
print('Create the features and labels manually:')
X = df['title']
y = df['target']
# Split the data into training and testing sets
print('Split the data into training and testing sets:')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create the vocabulary
print('Create the vocabulary:')
corpus = X_train
vocabulary = {}
for doc in corpus:
    for word in doc.split():
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
# Create the tfidf matrix
print('Create the tfidf matrix:')
tfidf_train = np.zeros((len(corpus), len(vocabulary)))
for i, doc in enumerate(corpus):
    for word in doc.split():
        tfidf_train[i, vocabulary[word]] += 1
# Create the tfidf matrix
print('Create the tfidf matrix:')
tfidf_test = np.zeros((len(X_test), len(vocabulary)))
for i, doc in enumerate(X_test):
    for word in doc.split():
        tfidf_test[i, vocabulary[word]] += 1
# Initialize the KNN classifier
print('Initialize the KNN classifier:')
knn = KNeighborsClassifier(n_neighbors=10)
# Fit the model
print('Fit the model:')
knn.fit(tfidf_train, y_train)
# Predict on the test set
print('Predict on the test set:')
y_pred = knn.predict(tfidf_test)
# Print the accuracy score
print('Print the accuracy score:')
print('Accuracy score: {}'.format(accuracy_score(y_test, y_pred)))
# Print the confusion matrix
print('Print the confusion matrix:')
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
# Print the classification report
print('Print the classification report:')
print('Classification report:')
print(classification_report(y_test, y_pred))