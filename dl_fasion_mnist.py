import tensorflow as tf
from tensorflow import keras
import utils_fastion_mnist


module_name = 'DL'

((x_train, y_train), (x_test, y_test)) = utils_fastion_mnist.load_fashion_mnist_data()

print("---------  run " + module_name + "  -------------")
# Create the neural network model
# Optimized to 4 Layears
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.2)


y_pred_clf = model.predict_classes(x_test, verbose=0)

utils_fastion_mnist.generate_report(module_name, y_pred_clf, y_test)
