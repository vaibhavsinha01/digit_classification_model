# make a project with cnn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
import joblib

mnist = fetch_openml('mnist_784',version=1)

X,y = mnist['data'],mnist['target']
# some_digit = X[0]
# some_digit_image = some_digit.reshape(28,28)

# plt.imshow(some_digit_image,cmap='binary')
# plt.axis('off')
# plt.show()
print('data loaded')
x_train,x_test= X[:60000],X[60000:]
print('data splitted')
scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.fit(x_test)
print('data is fitted')

y_train,y_test = y[:60000],y[60000:]
y_test = y_test.astype(float)
y_train = y_train.astype(float)
print('y splitted')
# clf=RandomForestClassifier()
clf=XGBClassifier()

clf.fit(x_train,y_train)
print('data fitted')

y_pred = clf.predict(x_test)

print("data predicted")

m = confusion_matrix(y_pred,y_test)
p = precision_score(y_pred,y_test,average='weighted')
print(m)
print(f"The precision score is:{p}")

joblib.dump(clf,'image_classification_model')





