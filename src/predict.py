import os, signal
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

predict_data_dir = './predictions/data'
predict_dir = './predictions'
predict_data = os.listdir(predict_data_dir)
data_count = len(os.listdir(predict_data_dir))

print(predict_data[:10])

print(('total prediction images:', data_count))

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_pic = [os.path.join(predict_data_dir, fname)
                for fname in predict_data[pic_index-8:pic_index]]

for i, img_path in enumerate(next_pic):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# Load model
model = load_model('./my_model.h5')

model.summary()

predict_datagen = ImageDataGenerator(rescale=1./255)

# Flow validation images in batches of 20 using test_datagen generator
predict_generator = predict_datagen.flow_from_directory(
    predict_dir,
    target_size=(150, 150),
    class_mode='binary')

predictions = model.predict_generator(
    predict_generator,
    verbose=2)

print(predictions)

for fname, result in zip(predict_data, predictions):
    print(fname)
    print(result)
    print("Dog" if result[0] > .5 else "Cat")
