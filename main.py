import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Store status for ESP32
last_status = {"command": 0}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
try:
    model = tf.keras.models.load_model("tomato_pro_model.h5")
    print("✅ Tomato Brain connected!")
except Exception as e:
    print(f"❌ Error: {e}")

TREATMENTS = {
    "Tomato_Bacterial_spot": {"status": "BACTERIAL", "solution": "Apply copper-based fungicides."},
    "Tomato_Early_blight": {"status": "FUNGAL", "solution": "Prune lower leaves."},
    "Tomato_Late_blight": {"status": "FUNGAL", "solution": "Use fungicides immediately."},
    "Tomato_Leaf_Mold": {"status": "FUNGAL", "solution": "Reduce humidity."},
    "Tomato_Septoria_leaf_spot": {"status": "FUNGAL", "solution": "Remove infected foliage."},
    "Tomato_Spider_mites_Two-spotted_spider_mite": {"status": "PEST", "solution": "Use insecticidal soap."},
    "Tomato__Target_Spot": {"status": "FUNGAL", "solution": "Ensure wide plant spacing."},
    "Tomato__Tomato_Yellow_Leaf_Curl_virus": {"status": "VIRAL", "solution": "Control whiteflies."},
    "Tomato__Tomato_Mosaic_virus": {"status": "VIRAL", "solution": "Remove infected plants."},
    "Tomato_Healthy": {"status": "HEALTHY", "solution": "Your plant is in great shape!"}
}

class_names = list(TREATMENTS.keys())

@app.get("/esp32-check")
async def esp32_check():
    return last_status

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global last_status
    
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert('RGB').resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    idx = np.argmax(predictions[0])
    raw_name = class_names[idx]
    confidence = round(float(np.max(predictions[0]) * 100), 2)
    
    info = TREATMENTS.get(raw_name, {"status": "UNKNOWN", "solution": "Consult an expert."})
    clean_name = raw_name.replace("Tomato_", "").replace("_", " ").strip()

    # Update the ESP32 command
    last_status = {"command": 1 if info["status"] != "HEALTHY" else 0}

    return {
        "disease": clean_name.title(),
        "status": info["status"],
        "solution": info["solution"],
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
