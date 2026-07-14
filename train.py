import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


training_set = tf.keras.utils.image_dataset_from_directory(
   'train',
   labels="inferred",
   label_mode="categorical", #Binary (2 classes)
   class_names=None,
   color_mode="rgb",
   batch_size=32,
   image_size=(128, 128),
   shuffle=True,
   seed=None,
   validation_split=None,
   subset=None,
   interpolation="bilinear",
   follow_links=False,
   crop_to_aspect_ratio=False,
)


validation_set = tf.keras.utils.image_dataset_from_directory(
   'valid',
   labels="inferred",
   label_mode="categorical",
   class_names=None,
   color_mode="rgb",
   batch_size=32,
   image_size=(128, 128),
   shuffle=True,
   seed=None,
   validation_split=None,
   subset=None,
   interpolation="bilinear",
   follow_links=False,
   crop_to_aspect_ratio=False,
)
