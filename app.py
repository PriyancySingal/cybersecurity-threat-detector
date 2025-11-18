import gradio as gr
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load

# Load model and preprocessing pipeline
model = tf.keras.models.load_model('cybersecurity_model.h5')
preprocessor = load('preprocessing_pipeline.joblib')

def generate_adversarial_examples(model, x, y_true, epsilon=0.1):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_true = tf.reshape(y_true, (x.shape[0], 1))

    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = model(x, training=False)
        loss = tf.keras.losses.binary_crossentropy(y_true, predictions)
        gradient = tape.gradient(loss, x)
        adversarial_example = x + epsilon * tf.sign(gradient)

    return adversarial_example.numpy()

def predict(sensor_data, vehicle_speed, network_traffic, sensor_type, sensor_status,
            vehicle_model, firmware_version, geofencing_status):

    try:
        # Create dataframe
        df = pd.DataFrame([[sensor_data, vehicle_speed, network_traffic, sensor_type,
                            sensor_status, vehicle_model, firmware_version, geofencing_status]],
                          columns=['Sensor_Data', 'Vehicle_Speed', 'Network_Traffic',
                                   'Sensor_Type', 'Sensor_Status', 'Vehicle_Model',
                                   'Firmware_Version', 'Geofencing_Status'])

        # Preprocess
        processed = preprocessor.transform(df)
        processed = np.reshape(processed, (1, -1))

        # Adversarial example
        y_true = np.array([1])
        adv = generate_adversarial_examples(model, processed, y_true)
        adv = np.reshape(adv, (1, -1))

        # Predict
        pred = model.predict(adv)[0][0]

        if pred > 0.5:
            return f"🚨 High Probability of Attack ({pred:.4f})"
        else:
            return f"✔ Low Probability of Attack ({pred:.4f})"

    except Exception as e:
        return f"Error: {e}"

# Build UI
ui = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 100, label="Sensor Data"),
        gr.Slider(0, 200, label="Vehicle Speed (km/h)"),
        gr.Slider(0, 1000, label="Network Traffic (MB)"),
        gr.Dropdown(["Type 1", "Type 2", "Type 3"], label="Sensor Type"),
        gr.Dropdown(["Active", "Inactive", "Error"], label="Sensor Status"),
        gr.Dropdown(["Model A", "Model B", "Model C"], label="Vehicle Model"),
        gr.Dropdown(["v1.0", "v2.0", "v3.0"], label="Firmware Version"),
        gr.Dropdown(["Enabled", "Disabled"], label="Geofencing Status"),
    ],
    outputs="text",
    title="Cybersecurity Threat Prediction",
    description="Enter vehicle & sensor data to detect adversarial attacks."
)

ui.launch()
