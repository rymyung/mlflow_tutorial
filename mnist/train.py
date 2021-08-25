import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
trainX = train_X.reshape((train_X.shape[0], 28, 28, 1))
testX = test_X.reshape((test_X.shape[0], 28, 28, 1))
trainY = tf.keras.utils.to_categorical(train_Y)
testY = tf.keras.utils.to_categorical(test_Y)

model = tf.keras.models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1, batch_size=32, validation_data=(testX, testY))

signature = infer_signature(testX, model.predict(testX))
mlflow.keras.log_model(model, "mnist_cnn", signature=signature)