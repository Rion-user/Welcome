import tensorflow as tf
import pandas as pd

파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
보스턴 = pd.read_csv(파일경로)

독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox',
             'rm', 'age', 'dis', 'rad', 'tax',
             'ptratio','b', 'lstat']]
종속 = 보스턴['medv']

X = tf.keras.layers.Input(shape=[13])
H = tf.keras.layers.Dense(10, activation='swish')(X)
H = tf.keras.layers.Dense(10, activation='swish')(H)
Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X,Y)
model.compile(loss = 'mse',optimizer = 'sgd', metrics = ['accuracy'])

model.fit(독립,종속,epochs = 10000, verbose = 0)
model.fit(독립,종속,epochs = 10)
