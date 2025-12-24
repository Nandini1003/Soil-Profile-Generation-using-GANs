from flask import Flask, render_template, request
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import os

# -------------------------------------------------
# Flask App Initialization
# -------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# Load Models & Artifacts
# -------------------------------------------------
try:
    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "classification_model (1).h5"),
        compile=False
    )

    generator_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "generator_epoch_5000 (1).h5"),
        compile=False
    )

    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

    print("✅ Models loaded successfully")

except Exception as e:
    print("❌ Error loading models:", e)
    raise RuntimeError("Model loading failed")

# -------------------------------------------------
# Home Page
# -------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# -------------------------------------------------
# Generate Soil Profile (GAN)
# -------------------------------------------------
@app.route("/generate", methods=["POST"])
def generate_new_profiles():
    try:
        input_dim = 9
        noise = np.random.rand(1, input_dim)

        generated = generator_model.predict(noise, verbose=0)
        generated_original = scaler.inverse_transform(generated)

        generated_values = generated_original[0].tolist()

        return render_template(
            "index.html",
            generated_values=generated_values
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Generation Error: {str(e)}"
        )

# -------------------------------------------------
# Classify & Evaluate Soil
# -------------------------------------------------
@app.route("/classify", methods=["POST"])
def classify_and_evaluate():
    try:
        form = request.form

        values = [
            float(form["pH"]),
            float(form["soil_ec"]),
            float(form["phosphorus"]),
            float(form["potassium"]),
            float(form["urea"]),
            float(form["tsp"]),
            float(form["mop"]),
            float(form["moisture"]),
            float(form["temperature"]),
        ]

        columns = [
            "pH", "Soil EC", "Phosphorus", "Potassium",
            "Urea", "T.S.P", "M.O.P", "Moisture", "Temperature"
        ]

        input_df = pd.DataFrame([values], columns=columns)
        scaled_input = scaler.transform(input_df)

        preds = model.predict(scaled_input, verbose=0)
        predicted_class = np.argmax(preds, axis=1)[0]
        plant_type = label_encoder.inverse_transform([predicted_class])[0]

        evaluation, conclusion = evaluate_soil(*values)

        return render_template(
            "index.html",
            plant_type=plant_type,
            evaluation=evaluation,
            conclusion=conclusion
        )

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numeric values."
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Classification Error: {str(e)}"
        )

# -------------------------------------------------
# Soil Evaluation Logic
# -------------------------------------------------
def evaluate_soil(pH, soil_ec, phosphorus, potassium, urea, tsp, mop, moisture, temperature):
    evaluation = {
        "pH": "Good" if 6.0 <= pH <= 7.5 else "Bad",
        "Soil EC": "Good" if 0.1 <= soil_ec <= 0.5 else "Bad",
        "Phosphorus": "Good" if 10 <= phosphorus <= 40 else "Bad",
        "Potassium": "Good" if 120 <= potassium <= 200 else "Bad",
        "Urea": "Good" if 20 <= urea <= 50 else "Bad",
        "T.S.P": "Good" if 10 <= tsp <= 30 else "Bad",
        "M.O.P": "Good" if 20 <= mop <= 60 else "Bad",
        "Moisture": "Good" if 60 <= moisture <= 80 else "Bad",
        "Temperature": "Good" if 15 <= temperature <= 35 else "Bad",
    }

    bad_params = [k for k, v in evaluation.items() if v == "Bad"]

    if not bad_params:
        conclusion = "This soil is ideal for plant growth."
    elif len(bad_params) <= 2:
        conclusion = "Mostly good soil. Improve: " + ", ".join(bad_params)
    else:
        conclusion = "Soil quality is poor. Critical issues in: " + ", ".join(bad_params)

    return evaluation, conclusion

# -------------------------------------------------
# Run Server
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
