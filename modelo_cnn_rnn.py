import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def crear_modelo():
    modelo = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(None, 128, 128, 3)),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(50),
        Dense(3, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

def generate_sequences(directory, batch_size, seq_length, img_height, img_width, subset):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(img_height, img_width),
        batch_size=batch_size * seq_length,
        class_mode='categorical',
        subset=subset
    )
    while True:
        batch_x, batch_y = next(generator)
        # Verificar si hay suficientes imágenes para llenar el lote completamente
        if batch_x.shape[0] == batch_size * seq_length:
            batch_x = np.reshape(batch_x, (batch_size, seq_length, img_height, img_width, 3))
            batch_y = batch_y[::seq_length]  # Tomar la etiqueta de la primera imagen de cada secuencia
            yield batch_x, batch_y
        else:
            # Ignorar este lote si no hay suficientes imágenes para formar secuencias completas
            continue

def preparar_datos(directorio_datos, batch_size, seq_length):
    generador_entrenamiento = generate_sequences(directorio_datos, batch_size, seq_length, 128, 128, 'training')
    generador_validacion = generate_sequences(directorio_datos, batch_size, seq_length, 128, 128, 'validation')
    return generador_entrenamiento, generador_validacion

def entrenar_modelo(generador_entrenamiento, generador_validacion, steps_per_epoch, validation_steps, epochs=10):
    modelo = crear_modelo()
    historial = modelo.fit(
        generador_entrenamiento,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=generador_validacion,
        validation_steps=validation_steps
    )
    return modelo, historial

# Configuración de los parámetros del modelo y entrenamiento
directorio_datos = 'D:\\TalentoTech\\modeloCNN_RNN\\imgPaneles'  # Asegúrate de que la ruta sea accesible
batch_size = 5
seq_length = 5
steps_per_epoch = 100   # Número de veces que se procesará el generador de entrenamiento en una época
validation_steps = 50   # Número de veces que se procesará el generador de validación en una época

generador_entrenamiento, generador_validacion = preparar_datos(directorio_datos, batch_size, seq_length)
modelo, historial = entrenar_modelo(generador_entrenamiento, generador_validacion, steps_per_epoch, validation_steps)