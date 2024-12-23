import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Memeriksa bentuk data sebelum mengonversinya menjadi array numpy
for i, item in enumerate(data_dict['data']):
    if len(item) != 42:  # Asumsikan panjang yang diharapkan adalah 42
        print(f'Error at index {i}: Shape {np.shape(item)}')
        # Anda bisa menambahkan logika untuk menormalkan bentuk data di sini

# Mengonversi data menjadi array numpy setelah bentuknya seragam
data = np.asarray([np.resize(item, (42,)) for item in data_dict['data']])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Menyimpan model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
