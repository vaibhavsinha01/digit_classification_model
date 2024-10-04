# make a project with cnn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import numpy as np

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
model=Sequential()
model.add(Dense(1568,input_dim = 784,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1568,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=15,validation_data=(x_test,y_test))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

score_acc = accuracy_score(y_test, y_pred_classes)
score_pre = precision_score(y_test, y_pred_classes, average='weighted')

print('The accuracy and precision scores are:')
print(score_acc)
print(score_pre)

# Save the model
model.save('image_number_classification_model.h5')
