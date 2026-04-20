import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# SECURITY: Allows your website to talk to this Python script
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the Model
try:
    model = tf.keras.models.load_model("tomato_pro_model.h5")
    print("✅ Tomato Brain connected successfully!")
except Exception as e:
    print(f"❌ Error: Could not find tomato_pro_model.h5 in this folder. {e}")

# 2. Knowledge Base (Matches standard PlantVillage dataset labels)
TREATMENTS = {
    "Tomato_Bacterial_spot": {"status": "BACTERIAL", "solution": "Apply copper-based fungicides. Avoid overhead watering to stop the spread."},
    "Tomato_Early_blight": {"status": "FUNGAL", "solution": "Prune lower leaves for airflow. Apply organic Neem oil or sulfur sprays."},
    "Tomato_Late_blight": {"status": "FUNGAL", "solution": "Destroy infected plants. Use fungicides containing chlorothalonil immediately."},
    "Tomato_Leaf_Mold": {"status": "FUNGAL", "solution": "Reduce humidity and improve air circulation in the growing area."},
    "Tomato_Septoria_leaf_spot": {"status": "FUNGAL", "solution": "Remove infected foliage and clear soil debris. Apply copper spray."},
    "Tomato_Spider_mites_Two-spotted_spider_mite": {"status": "PEST", "solution": "Spray leaves with a strong water stream or use insecticidal soap."},
    "Tomato__Target_Spot": {"status": "FUNGAL", "solution": "Maintain dry foliage and ensure wide plant spacing for airflow."},
    "Tomato__Tomato_Yellow_Leaf_Curl_virus": {"status": "VIRAL", "solution": "Control whiteflies with sticky traps. Remove infected plants."},
    "Tomato__Tomato_Mosaic_virus": {"status": "VIRAL", "solution": "No cure. Remove and burn infected plants to prevent spread."},
    "Tomato_Healthy": {"status": "HEALTHY", "solution": "Your plant is in great shape! Continue regular watering and care."}
}

class_names = list(TREATMENTS.keys())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Process Image
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert('RGB').resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # AI Prediction
    predictions = model.predict(img_array)
    idx = np.argmax(predictions[0])
    raw_name = class_names[idx]
    confidence = round(float(np.max(predictions[0]) * 100), 2)
    
    # Get details from Knowledge Base
    info = TREATMENTS.get(raw_name, {"status": "UNKNOWN", "solution": "Consult an expert."})
    
    # Clean name for UI (e.g. "Tomato_Bacterial_spot" -> "Bacterial Spot")
    clean_name = raw_name.replace("Tomato_", "").replace("_", " ").replace("  ", " ").strip()

    return {
        "disease": clean_name.title(),
        "status": info["status"],
        "solution": info["solution"],
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # Create a storage spot for the ESP32
last_status = {"command": 0}

@app.get("/esp32-check")
async def esp32_check():
    return last_status

# Update your existing predict function to save the result
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ... (keep all your existing image processing code) ...
    
    # ADD THIS LINE right before the 'return' statement:
    global last_status
    last_status = {"command": 1 if info["status"] != "HEALTHY" else 0}
    
    return {
        "disease": clean_name.title(),
        "status": info["status"],
        "solution": info["solution"],
        "confidence": confidence
    }
