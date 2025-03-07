import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split  # Untuk membagi dataset menjadi data latih dan uji
from tensorflow.keras.models import Sequential  # Untuk membuat model sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization  # Layer yang digunakan dalam CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Untuk augmentasi dan preprocessing data gambar
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # Untuk menghentikan training lebih awal jika tidak ada peningkatan
from tensorflow.keras.optimizers import Adam  # Optimizer yang digunakan

# Direktori dataset
train_dir = 'images/train/'  # Path ke folder gambar training
test_dir = 'images/test/'  # Path ke folder gambar testing

# Membuat ImageDataGenerator untuk augmentasi data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisasi nilai piksel (0-255 menjadi 0-1)
    rotation_range=20,  # Rotasi gambar secara acak hingga 20 derajat
    width_shift_range=0.2,  # Pergeseran horizontal acak hingga 20%
    height_shift_range=0.2,  # Pergeseran vertikal acak hingga 20%
    shear_range=0.2,  # Distorsi bentuk gambar secara acak
    zoom_range=0.2,  # Zoom gambar hingga 20%
    horizontal_flip=True,  # Membalik gambar secara horizontal
    fill_mode='nearest'  # Mengisi area kosong akibat transformasi
)

# Generator data uji tanpa augmentasi, hanya normalisasi
test_datagen = ImageDataGenerator(rescale=1./255)

# Menggunakan flow_from_directory untuk memuat data langsung dari folder
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Path folder dataset training
    target_size=(150, 150),  # Ukuran gambar yang akan digunakan
    batch_size=32,  # Jumlah gambar per batch
    class_mode='categorical'  # Mode klasifikasi multi-kelas (one-hot encoding)
)

test_generator = test_datagen.flow_from_directory(
    test_dir,  # Path folder dataset testing
    target_size=(150, 150),  # Ukuran gambar yang akan digunakan
    batch_size=32,  # Jumlah gambar per batch
    class_mode='categorical'  # Mode klasifikasi multi-kelas (one-hot encoding)
)

# =======================================================================
#                Jika Langsung Menggunakan Dataframe 
# =======================================================================
# Membaca gambar dari DataFrame untuk data latih
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,  # DataFrame yang digunakan (train set)
    directory='images/',  # Folder tempat gambar disimpan
    x_col='filename',  # Nama file gambar
    y_col='label',  # Label kelas gambar
    target_size=(150, 150),  # Ukuran gambar diubah menjadi 150x150 piksel
    batch_size=32,  # Jumlah gambar yang diproses dalam satu batch
    class_mode='categorical'  # Label dikonversi ke one-hot encoding
)
# =======================================================================

# Membangun model CNN secara berurutan
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),  # Layer konvolusi pertama dengan 32 filter
    # input_shape menentukan dimensi input gambar:
    # - Nilai pertama (150) adalah tinggi gambar yang bisa diubah sesuai dataset
    # - Nilai kedua (150) adalah lebar gambar yang bisa diubah sesuai dataset
    # - Nilai ketiga (3) adalah jumlah channel warna (3 untuk RGB, 1 untuk grayscale)
    MaxPooling2D(2,2),  # Pooling layer pertama untuk mengurangi dimensi fitur
    Conv2D(64, (3,3), activation='relu'),  # Layer konvolusi kedua dengan 64 filter
    MaxPooling2D(2,2),  # Pooling layer kedua
    Conv2D(128, (3,3), activation='relu'),  # Layer konvolusi ketiga dengan 128 filter
    MaxPooling2D(2,2),  # Pooling layer ketiga
    Flatten(),  # Meratakan hasil ekstraksi fitur menjadi vektor 1D
    Dense(512, activation='relu'),  # Fully connected layer dengan 512 neuron
    Dropout(0.5),  # Dropout dengan rate 0.5 untuk mencegah overfitting
    Dense(train_generator.num_classes, activation='softmax')  # Layer output dengan jumlah neuron sesuai jumlah kelas
])

# Kompilasi model dengan loss function, optimizer, dan metrik evaluasi
model.compile(
    loss='categorical_crossentropy',  # Fungsi loss untuk klasifikasi multi-kelas
    optimizer=Adam(learning_rate=0.001),  # Optimizer Adam dengan learning rate 0.001
    metrics=['accuracy']  # Metrik evaluasi menggunakan akurasi
)

# Membuat callback untuk early stopping dan menyimpan model terbaik
early_stopping = EarlyStopping(
    monitor='val_loss',  # Menghentikan training jika val_loss tidak membaik
    patience=5,  # Toleransi epoch tanpa peningkatan sebelum dihentikan
    restore_best_weights=True  # Menggunakan bobot terbaik setelah training dihentikan
)
model_checkpoint = ModelCheckpoint(
    'best_model.h5',  # Nama file model yang disimpan
    monitor='val_loss',  # Simpan model jika val_loss membaik
    save_best_only=True  # Hanya menyimpan model terbaik
)

# Melatih model dengan dataset yang telah diproses
history = model.fit(
    train_generator,  # Data training
    validation_data=test_generator,  # Data validasi (test set)
    epochs=20,  # Jumlah epoch training
    callbacks=[early_stopping, model_checkpoint]  # Menggunakan callback early stopping dan checkpoint
)

# Menampilkan grafik akurasi model
plt.plot(history.history['accuracy'], label='Training Accuracy')  # Plot akurasi training
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot akurasi validasi
plt.xlabel('Epochs')  # Label sumbu X
plt.ylabel('Accuracy')  # Label sumbu Y
plt.legend()  # Menampilkan legenda grafik
plt.show()

# Menyimpan model akhir setelah training selesai
model.save('final_model.h5')  # Menyimpan model akhir ke dalam file
