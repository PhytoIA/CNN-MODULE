import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import seaborn as sns
import streamlit as st
import librosa


print("\n--- TOUS LES PACKAGES SONT PRÊTS ---")
print("TensorFlow:", tf.__version__, "| GPU M4 actif:", len(tf.config.list_physical_devices('GPU')) > 0)
print("Pandas:", pd.__version__)
print("Numpy:", np.__version__)
print("Librosa (Audio) et Streamlit sont bien chargés !")
