import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import os

X, y = [], []

# Letters klasöründeki tüm *_data.csv dosyalarını bul
for file in glob.glob(os.path.join("Letters", "*_data.csv")):
    label = os.path.basename(file).split('_')[0].upper()  # Dosya adı başındaki harf veya kelime
    data = np.loadtxt(file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    X.append(data)
    y += [label] * data.shape[0]

X = np.vstack(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)
print("Test accuracy:", clf.score(X_test, y_test))
joblib.dump(clf, "hand_sign_svm.pkl")