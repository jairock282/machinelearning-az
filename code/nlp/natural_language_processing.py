import pandas as pd
import re  # Regular expressions
import nltk
nltk.download("stopwords")  # Download dataset of irrelevant words
from nltk.corpus import stopwords  # Use the corpus already downloaded
from nltk.stem.porter import PorterStemmer  #Reduce any word to its root(infinitive)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Import dataset
dataset = pd.read_csv("/mnt/SSD2/linux/Documents/cursos/machine_learning_A-Z/machinelearning-az/datasets/Part 7 - "
                      "Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv",
                      delimiter="\t", # Set tab as delimiter
                      quoting=3)  # Ignore double quotes

corpus = []
# Data cleaning
for i in range(0, dataset.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', dataset["Review"][i])  # Removes special characters
    review = review.lower()  # To lower case
    review = review.split()

    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]  # Set language, we used a set in order to improve speed

    review = ' '.join(review)

    corpus.append(review)

# Create Bag Of Words
# Create a column for each different word, and count theirs instances (Disperse Matrix)

# Tokenization
cv = CountVectorizer(max_features=1500)  # Word2Vect, Here we translate the word problem into a number problem
X = cv.fit_transform(corpus).toarray()  #fit - apply function, tranform - create matrix
y = dataset.iloc[:, 1].values

# Classification algorithm
#  Splits dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# ===========================================================================
# Train model
# Naive Bayes
classifier_naive_bayes = GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)

# Prediction
y_pred = classifier_naive_bayes.predict(X_test)

# Evaluates model (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

print("\n===== Naive Bayes =====")
print(f"acc: {accuracy}")
print(f"pre: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")
# ===========================================================================
#Random forest
classifier_random_forest = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
classifier_random_forest.fit(X_train, y_train)

# Prediction
y_pred = classifier_random_forest.predict(X_test)

# Evaluates model (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

print("\n===== Random forest =====")
print(f"acc: {accuracy}")
print(f"pre: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")
# ===========================================================================
#KNN
# Scales values
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier_knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier_knn.fit(X_train, y_train)

# Prediction
y_pred = classifier_knn.predict(X_test)

# Evaluates model (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

print("\n===== KNN =====")
print(f"acc: {accuracy}")
print(f"pre: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")
# ===========================================================================
# Logistic regression

# Scales values
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Train model
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(X_train, y_train)

# Prediction
y_pred = classifier_lr.predict(X_test)

# Evaluates model (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

print("\n===== Logistic regression =====")
print(f"acc: {accuracy}")
print(f"pre: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")
# ===========================================================================
# SVM

# Scales values
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Train model
classifier_svc = SVC(kernel="linear", random_state=0)
classifier_svc.fit(X_train, y_train)

# Prediction
y_pred = classifier_svc.predict(X_test)

# Evaluates model (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

print("\n===== SVM =====")
print(f"acc: {accuracy}")
print(f"pre: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")

# ===========================================================================
# Decision Tree
classifier_tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier_tree.fit(X_train, y_train)

# Prediction
y_pred = classifier_tree.predict(X_test)

# Evaluates model (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

print("\n===== Decision tree =====")
print(f"acc: {accuracy}")
print(f"pre: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")

# ===========================================================================
# CART
classifier_cart = DecisionTreeClassifier(criterion="gini", random_state=0)
classifier_cart.fit(X_train, y_train)

# Prediction
y_pred = classifier_cart.predict(X_test)

# Evaluates model (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

print("\n===== CART =====")
print(f"acc: {accuracy}")
print(f"pre: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")

