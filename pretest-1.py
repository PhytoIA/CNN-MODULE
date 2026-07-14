import tensorflow as tf


print("\n--- TEST SUR FICHIER RÉUSSI ---")
print("Version installée :", tf.__version__)
print("Est-ce que le GPU M4 fonctionne ?", len(tf.config.list_physical_devices('GPU')) > 0)
print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('CPU'))


# Un mini calcul rapide pour forcer le GPU à travailler
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[3.0], [4.0]])
print("Résultat du calcul :", tf.matmul(a, b).numpy()[0][0])
