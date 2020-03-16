import pandas as pd
import csv
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# Read CSV file into DataFrame
df_amazon = pd.read_csv('amazon_cells_labelled.txt', sep='\t', header=None)
df_yelp = pd.read_csv('yelp_labelled.txt', sep='\t', header=None)
# Setting `quoting=csv.QUOTE_NONE` because there are quotes in this file,
# otherwise some data will be missing
df_imdb = pd.read_csv('imdb_labelled.txt', sep='\t', quoting=csv.QUOTE_NONE, header=None)


# print the shape of each dataframe to avoid missing data
# print(df_amazon.shape)
# print(df_imdb.shape)
# print(df_yelp.shape)

# Concatenate three dataframes into one
df = pd.concat([df_amazon, df_yelp, df_imdb], ignore_index=True)
# print(df.shape)

# Predictor and response vector
X = df.iloc[:, 0]
y = df.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Using CountVectorizer to convert a collection of text documents to a matrix of token counts
# Defaulted configuration without removing stop words
count_vect = CountVectorizer()
X_count_train = count_vect.fit_transform(X_train)
X_count_test = count_vect.transform(X_test)

# Configuration with removing stop words
count_stop_vect = CountVectorizer(analyzer='word', stop_words='english')
X_count_stop_train = count_stop_vect.fit_transform(X_train)
X_count_stop_test = count_stop_vect.transform(X_test)

# Using TfidfVectorizer to extract feature vector
# Defaulted configuration without removing stop words
tfid_stop_vec = TfidfVectorizer()
X_tfid_train = tfid_stop_vec.fit_transform(X_train)
X_tfid_test = tfid_stop_vec.transform(X_test)

# Configuration with removing stop words
tfid_stop_vec = TfidfVectorizer(analyzer='word', stop_words='english')
X_tfid_stop_train = tfid_stop_vec.fit_transform(X_train)
X_tfid_stop_test = tfid_stop_vec.transform(X_test)


# Using Naive Bayes Classifier, learn and predict the two extracted feature values ​​separately
# Defaulted CountVectorizer
mnb_count = MultinomialNB()
mnb_count.fit(X_count_train, y_train)
mnb_count_y_predict = mnb_count.predict(X_count_test)
mnb_count_y_predict_proba = mnb_count.predict_proba(X_count_test)[:, 1]
auc_count = roc_auc_score(y_test, mnb_count_y_predict_proba)


# CountVectorizer with removing stop words
mnb_count_stop = MultinomialNB()
mnb_count_stop.fit(X_count_stop_train, y_train)
mnb_count_stop_y_predict = mnb_count_stop.predict(X_count_stop_test)
mnb_count_stop_y_predict_proba = mnb_count_stop.predict_proba(X_count_stop_test)[:, 1]
auc_count_stop = roc_auc_score(y_test, mnb_count_stop_y_predict_proba)


# Defaulted TfidfVectorizer
mnb_tfid = MultinomialNB()
mnb_tfid.fit(X_tfid_train, y_train)
mnb_tfid_y_predict = mnb_tfid.predict(X_tfid_test)
mnb_tfid_y_predict_proba = mnb_tfid.predict_proba(X_tfid_test)[:, 1]
auc_tfid = roc_auc_score(y_test, mnb_tfid_y_predict_proba)



# TfidfVectorizer with removing stop words
mnb_tfid_stop = MultinomialNB()
mnb_tfid_stop.fit(X_tfid_stop_train, y_train)
mnb_tfid_stop_y_predict = mnb_tfid_stop.predict(X_tfid_stop_test)
mnb_tfid_stop_y_predict_proba = mnb_count_stop.predict_proba(X_tfid_stop_test)[:, 1]
auc_tfid_stop = roc_auc_score(y_test, mnb_tfid_stop_y_predict_proba)


# Model Evaluation(Naive Bayes)
# The Report consists of precision, recall, f1-score and support.
# Note that in binary classification, recall of the positive class is also known as “sensitivity”;
# recall of the negative class is “specificity”
print("**********************Model Evaluation(Naive Bayes)***********************")
print("Accuracy of defaulted CountVectorizer：%.3f" % (mnb_count.score(X_count_test, y_test)*100), "%")
print(f'Area Under Curve={auc_count * 100:.3f}%')
print("Report of defaulted CountVectorizer :\n", classification_report(mnb_count_y_predict, y_test))
print("**************************************************************************")
print("Accuracy of CountVectorizer with removing stop words：%.3f" % (mnb_count_stop.score(X_count_stop_test, y_test)*100) ,"%")
print(f'Area Under Curve={auc_count_stop * 100:.3f}%')
print("Report of CountVectorizer with removing stop words:\n", classification_report(mnb_count_stop_y_predict, y_test))
print("**************************************************************************")
print("Accuracy of defaulted TfidfVectorizer：%.3f" % (mnb_tfid.score(X_tfid_test, y_test)*100), "%")
print(f'Area Under Curve={auc_tfid * 100:.3f}%')
print("Report of defaulted TfidfVectorizer:\n", classification_report(mnb_tfid_y_predict, y_test))
print("**************************************************************************")
print("Accuracy of TfidfVectorizer with removing stop words：%.3f" % (mnb_tfid_stop.score(X_tfid_stop_test, y_test)*100),"%")
print(f'Area Under Curve={auc_tfid_stop * 100:.3f}%')
print("Report of TfidfVectorizer with removing stop words:\n", classification_report(mnb_tfid_stop_y_predict, y_test))


SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

SVM.fit(X_count_train, y_train)
predictions_SVM_count = SVM.predict(X_count_test)
decision_scores_count = SVM.decision_function(X_count_test)


SVM.fit(X_count_stop_train, y_train)
predictions_SVM_count_stop = SVM.predict(X_count_stop_test)
decision_scores_count_stop = SVM.decision_function(X_count_stop_test)


SVM.fit(X_tfid_train, y_train)
predictions_SVM_tfid = SVM.predict(X_tfid_test)
decision_scores_tfid = SVM.decision_function(X_tfid_test)


SVM.fit(X_tfid_stop_train, y_train)
predictions_SVM_tfid_stop = SVM.predict(X_tfid_stop_test)
decision_scores_tfid_stop = SVM.decision_function(X_tfid_stop_test)

# Model Evaluation(SVM)
print("**************************Model Evaluation(SVM)***************************")
print("Accuracy of defaulted CountVectorizer：%.3f " % (accuracy_score(predictions_SVM_count, y_test)*100), "%")
print(f'Area Under Curve={roc_auc_score(y_test, decision_scores_count) * 100:.3f}%')
print("Report of defaulted CountVectorizer :\n", classification_report(predictions_SVM_count, y_test))
print("**************************************************************************")
print("Accuracy of CountVectorizer with removing stop words：%.3f " % (accuracy_score(predictions_SVM_count_stop, y_test)*100), "%")
print(f'Area Under Curve={roc_auc_score(y_test, decision_scores_count_stop) * 100:.3f}%')
print("Report of CountVectorizer with removing stop words:\n", classification_report(predictions_SVM_count_stop, y_test))
print("**************************************************************************")
print("Accuracy of defaulted TfidfVectorizer：%.3f" % (accuracy_score(predictions_SVM_tfid, y_test)*100), "%")
print(f'Area Under Curve={roc_auc_score(y_test, decision_scores_tfid) * 100:.3f}%')
print("Report of defaulted TfidfVectorizer:\n", classification_report(predictions_SVM_tfid, y_test))
print("**************************************************************************")
print("Accuracy of TfidfVectorizer with removing stop words：%.3f " % (accuracy_score(predictions_SVM_tfid_stop, y_test)*100), "%")
print(f'Area Under Curve={roc_auc_score(y_test, decision_scores_tfid_stop) * 100:.3f}%')
print("Report of TfidfVectorizer with removing stop words:\n", classification_report(predictions_SVM_tfid_stop, y_test))

