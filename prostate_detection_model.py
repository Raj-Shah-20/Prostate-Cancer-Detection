from xml.etree.ElementPath import xpath_tokenizer
import pandas as pd
import numpy as np
import pickle
import pandas as pd

data= pd.read_csv('Prostate_Cancer.csv')

X=data
y=data

# print(X)
# print(y)
# print(X.isna().sum())

X_aug = pd.DataFrame(X)
data = pd.concat([X_aug]*5, ignore_index=True)
# print(data)

X=data.drop(data.columns[[0,1]], axis = 1)
y=data['diagnosis_result']
# print(X)
# print(y)

# print(X.isna().any())
# print(y.isna().any())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# print(len(X_train))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sc = scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
std  = np.sqrt(sc.var_)
np.save('std.npy',std )
np.save('mean.npy',sc.mean_)
# print(scale1.mean_)

# from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score
from xgboost import XGBClassifier
classifier = XGBClassifier(use_label_encoder=False, eval_metric= 'error')
classifier.fit(X_train, y_train)
y_pred= classifier.predict(X_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))
print(recall_score(y_test, y_pred))


filename = 'finalized_model.sav'
pickle.dump(classifier,open(filename,'wb'))
# print(X_test)