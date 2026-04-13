from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape, num_classes):
    model = keras.Sequential(name="Classification_Model")

    model.add(layers.Conv1D(filters=32, kernel_size=64, strides=16, activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    
    model.add(layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    
    model.add(layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model