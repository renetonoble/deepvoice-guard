import io
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
MODEL_PATH = "deepfake_audio_detector.h5"
SAMPLE_RATE = 22050
DURATION = 2.0
N_MELS = 128
MAX_TIME_STEPS = 87

# Global model variable
model = None

# --- LIFECYCLE MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        print("Loading Model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load model. {e}")
    yield
    print("Shutting down...")

# --- INIT APP ---
app = FastAPI(title="DeepVoice Guard API", lifespan=lifespan)

# --- CORE LOGIC ---
def preprocess_audio(file_bytes: io.BytesIO):
    try:
        audio, _ = librosa.load(file_bytes, sr=SAMPLE_RATE, duration=DURATION)
        target_len = int(DURATION * SAMPLE_RATE)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        current_width = mel_spec_db.shape[1]
        if current_width < MAX_TIME_STEPS:
            padding = MAX_TIME_STEPS - current_width
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding)))
        else:
            mel_spec_db = mel_spec_db[:, :MAX_TIME_STEPS]

        return mel_spec_db.reshape(1, N_MELS, MAX_TIME_STEPS, 1)
    except Exception as e:
        print(f"Preprocessing Error: {e}")
        raise ValueError("Failed to process audio file")

@app.get("/")
async def root():
    return {"status": "DeepVoice Guard is Online", "gpu": len(tf.config.list_physical_devices('GPU')) > 0}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        contents = await file.read()
        audio_stream = io.BytesIO(contents)
        input_tensor = preprocess_audio(audio_stream)
        prediction = model.predict(input_tensor)
        confidence = float(prediction[0][0])
        is_fake = confidence > 0.5 
        return {
            "filename": file.filename,
            "label": "FAKE" if is_fake else "REAL",
            "confidence": confidence,
            "is_fake": is_fake
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
