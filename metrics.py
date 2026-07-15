import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

model = tf.keras.models.load_model("trained_model.keras")

training_set = tf.keras.utils.image_dataset_from_directory(
   'train',
   labels="inferred",
   label_mode="categorical", #Classes (38)
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

# Building Model

print(tf.config.list_physical_devices())

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.models import Sequential

# Some other metrics for model evaluation
class_name = validation_set.class_names

test_set = tf.keras.utils.image_dataset_from_directory(
   'valid',
   labels="inferred",
   label_mode="categorical", #Classes (38)
   class_names=None,
   color_mode="rgb",
   batch_size=32,
   image_size=(128, 128),
   shuffle=False,
   seed=None,
   validation_split=None,
   subset=None,
   interpolation="bilinear",
   follow_links=False,
   crop_to_aspect_ratio=False,
)

y_pred = model.predict(test_set)
y_pred,y_pred.shape

predicted_categories = tf.argmax(y_pred,axis=1)
true_categories = tf.concat([y for x,y in test_set], axis=0)

Y_true = tf.argmax(true_categories,axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(Y_true, predicted_categories, target_names=class_name))

cm = confusion_matrix(Y_true, predicted_categories)
cm.shape

# Confusion Matrix Visualization
plt.figure(figsize=(40,40))
sns.heatmap(cm, annot=True, annot_kws={'size': 10})
plt.xlabel("Predicted Class", fontsize=20)
plt.ylabel("Actual Class", fontsize=20)
plt.title("Plant Disease Prediction Confusion Matrix", fontsize=25)
plt.show()