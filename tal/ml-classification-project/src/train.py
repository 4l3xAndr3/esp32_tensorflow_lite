import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from model import create_yggdrasil_cnn_model

def load_data(data_dir):
    # Load processed data from the specified directory
    # This is a placeholder for actual data loading logic
    # Assuming data is in numpy format for simplicity
    X = np.load(os.path.join(data_dir, 'X_processed.npy'))
    y = np.load(os.path.join(data_dir, 'y_processed.npy'))
    return X, y

def train_model(X, y, input_shape, num_classes):
    model = create_yggdrasil_cnn_model(input_shape, num_classes)
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    return model

def save_model(model, model_path):
    model.save(model_path)

def main():
    data_dir = '../data/processed'
    model_path = '../models/yggdrasil_model.h5'
    
    X, y = load_data(data_dir)
    input_shape = (X.shape[1], 1)  # Assuming X is shaped (samples, 1024, 1)
    num_classes = len(np.unique(y))
    
    model = train_model(X, y, input_shape, num_classes)
    save_model(model, model_path)
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    main()