import tensorflow as tf
from tensorflow.keras import layers
import pathlib
import numpy as np
import os

def generate_model():
    print("===========================================")
    print("  ENTRAÎNEMENT DU MODÈLE DE FLEURS (INT8) ")
    print("===========================================")
    print("1. Téléchargement du dataset tf_flowers...")
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    batch_size = 32
    img_height = 96
    img_width = 96

    print("2. Préparation des données d'entraînement (96x96 Niveaux de gris)...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      color_mode="grayscale",
      image_size=(img_height, img_width),
      batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      color_mode="grayscale",
      image_size=(img_height, img_width),
      batch_size=batch_size)

    class_names = train_ds.class_names
    print(f"Classes détectées: {class_names}")
    num_classes = len(class_names)

    # Optimisation des performances pour le chargement
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("3. Création de l'architecture CNN (adaptée pour l'ESP32)...")
    model = tf.keras.Sequential([
      layers.Rescaling(1./127.5, offset=-1, input_shape=(img_height, img_width, 1)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("4. Début de l'entraînement (environ 15 époques)...")
    epochs = 15
    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    print("5. Entraînement terminé. Conversion en TensorFlow Lite INT8 (Quantization)...")

    # The dataset provides [0, 255] float/int inputs. We feed these for representative dataset.
    def representative_data_gen():
        # Retrieve the dataset without batching/shuffle optimization for pure raw sampling
        rep_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, color_mode="grayscale", image_size=(96, 96), batch_size=1, seed=123
        )
        for input_value, _ in rep_ds.take(100):
            yield [tf.cast(input_value, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Obliger TFLite à n'utiliser que des opérations INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Spécifier que l'entrée et la sortie seront en int8 [-128, 127]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Sauvegarder le modèle TFLite "pur" (utile pour vérifier avec Netron par exemple)
    tflite_filename = "flower_model_quantized.tflite"
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
    print(f"Modèle TFLite quantifié sauvegardé sous {tflite_filename} ({len(tflite_model)} octets).")

    print("6. Génération des fichiers C++ (flower_detect_model_data.cpp/.h)...")
    generate_c_array(tflite_model, "flower_detect_model_data.cpp", "g_flower_detect_model_data")
    print(">>> TOUT EST TERMINÉ ! <<<")
    print("Veuillez transférer les deux fichiers 'flower_detect_model_data.cpp' et '.h' à l'assistant pour intégration.")


def generate_c_array(tflite_model, filename, array_name):
    # Convertit un tflite binaire en tableau C hexadécimal
    hex_array = [hex(x) for x in tflite_model]
    
    hex_str = ""
    for i, val in enumerate(hex_array):
        hex_str += val + ", "
        if (i + 1) % 12 == 0:
            hex_str += "\n  "
            
    cpp_content = f"""// Automatiquement généré par train_flower_model.py
#include "flower_detect_model_data.h"

// Le modèle est stocké dans la mémoire flash (PROGMEM / alignas)
alignas(16) const unsigned char {array_name}[] = {{
  {hex_str}
}};

const int {array_name}_len = {len(tflite_model)};
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(cpp_content)

    h_content = f"""// Automatiquement généré
#ifndef FLOWER_DETECT_MODEL_DATA_H_
#define FLOWER_DETECT_MODEL_DATA_H_

extern const unsigned char {array_name}[];
extern const int {array_name}_len;

#endif
"""
    with open(filename.replace(".cpp", ".h"), "w", encoding="utf-8") as f:
        f.write(h_content)

if __name__ == "__main__":
    generate_model()
