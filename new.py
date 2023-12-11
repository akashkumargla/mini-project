import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist, emnist  # Import EMNIST dataset
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

# Load the EMNIST dataset
(x_train, y_train), (x_test, y_test) = emnist.load_data('letters')  # 'letters' for letters dataset, 'digits' for digits dataset

# Plot the first 10 samples
for i in range(10):
    plt.imshow(x_train[i], cmap='binary')
    plt.title(chr(y_train[i] + 96))  # Convert label to corresponding ASCII letter
    plt.show()

# Normalize and reshape the data
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train - 1, 26)  # Subtract 1 to map labels to [0, 25] range
y_test = keras.utils.to_categorical(y_test - 1, 26)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(26, activation="softmax"))  # 26 classes for letters

model.summary()

# Compile the model
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# Callbacks
es = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1)
mc = ModelCheckpoint("./bestmodel.h5", monitor="val_acc", verbose=1, save_best_only=True)
cb = [es, mc]

# Train the model
his = model.fit(x_train, y_train, epochs=2, validation_split=0.3, callbacks=cb)

# Load the saved model
model_S = keras.models.load_model('./bestmodel.h5')

# Evaluate the model on the test set
score = model_S.evaluate(x_test, y_test)
print(f"The model accuracy is {score[1]}")
