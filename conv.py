from __future__ import absolute_import, division, print_function, unicode_literals

import os

import PIL

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

class_names = ['hatsune miku', 'kinomoto sakura']

base_dir = './images/'

train_hm_dir = os.path.join(base_dir, 'hatsune')
train_ks_dir = os.path.join(base_dir, 'kinomoto')

validation_hm_dir = os.path.join(base_dir, 'hatsune')
validation_ks_dir = os.path.join(base_dir, 'kinomoto')

train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(base_dir,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
validation_generator = test_datagen.flow_from_directory(base_dir,
                                                        batch_size=10,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 2), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=20,
                              epochs=1,
                              validation_steps=20,
                              verbose=1)

model.predict()
