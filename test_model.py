import tensorflow as tf

model = tf.keras.models.load_model(
    "vgg16_best.keras",
    compile=False
)

print("SUCCESS")