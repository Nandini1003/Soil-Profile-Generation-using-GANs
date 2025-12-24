import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# -------------------------------
# 1. Create and save SCALER
# -------------------------------
X_dummy = np.random.rand(300, 9) * 100
scaler = StandardScaler()
scaler.fit(X_dummy)
joblib.dump(scaler, "scaler.pkl")

# -------------------------------
# 2. Create and save LABEL ENCODER
# -------------------------------
plant_labels = ["Rice", "Wheat", "Maize", "Cotton"]
label_encoder = LabelEncoder()
label_encoder.fit(plant_labels)
joblib.dump(label_encoder, "label_encoder.pkl")

# -------------------------------
# 3. Create CLASSIFICATION MODEL
# -------------------------------
classifier = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(9,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(plant_labels), activation="softmax")
])

classifier.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

y_dummy = np.random.randint(0, len(plant_labels), size=(300,))
classifier.fit(
    scaler.transform(X_dummy),
    y_dummy,
    epochs=5,
    verbose=1
)

classifier.save("classification_model (1).h5")

# -------------------------------
# 4. Create GENERATOR MODEL (GAN Generator)
# -------------------------------
generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(9,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(9, activation="linear")
])

generator.compile(optimizer="adam", loss="mse")
generator.save("generator_epoch_5000 (1).h5")

print("✅ All files created successfully")
