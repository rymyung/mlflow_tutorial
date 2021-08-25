import sys
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature


class mnistLearner():
    def __init__(self, loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'], **kwargs):
        
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        if 'epochs' in kwargs.keys():
            self.epochs = kwargs['epochs']
        else:
            self.epochs = 10

        if 'batch_size' in kwargs.keys():
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = 32


    def prepare_data(self):

        (train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()

        train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
        test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

        train_X = train_X.astype('float32')/255.
        test_X = test_X.astype('float32')/255.

        train_Y = tf.keras.utils.to_categorical(train_Y)
        test_Y = tf.keras.utils.to_categorical(test_Y)

        return train_X, test_X, train_Y, test_Y


    def set_optimizer(self, opt_name, lr=0.01):

        if opt_name == 'SGD':
            self.optimizer = SGD(lr)
        elif opt_name == 'RMSprop':
            self.optimizer = RMSprop(lr)


    def set_loss(self, loss_name):

        self.loss = loss_name
    

    def set_metrics(self, *args):
        
        self.metrics = list(*args)


    def build_model(self):

        self.model = tf.keras.models.Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        

    def learn_model(self, train_X, test_X, train_Y, test_Y):
        
        self.history = self.model.fit(train_X, train_Y, epochs=self.epochs, batch_size=self.batch_size, validation_data=(test_X, test_Y))

    
    def get_score(self, test_X, test_Y):

        score = self.model.evaluate(test_X, test_Y)

        return score


if __name__ == "__main__":

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    clf = mnistLearner(epochs=epochs, batch_size=batch_size)

    with mlflow.start_run():

        clf = mnistLearner(epochs=epochs, batch_size=batch_size)
        train_x, test_x, train_y, test_y = clf.prepare_data()
        clf.build_model()
        clf.learn_model(train_x, test_x, train_y, test_y)
        score = clf.get_score(test_x, test_y)

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("loss function", clf.loss)
        mlflow.log_param("optimizer", clf.optimizer)
        mlflow.log_metric("loss", score[0])
        mlflow.log_metric("accuracy", score[1])

        mlflow.keras.log_model(clf.model, "mnist_keras")