import io
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- CRITICAL IMPORT
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
MODEL_PATH = "model.tflite"
SAMPLE_RATE = 22050
DURATION = 2.0
N_MELS = 128
MAX_TIME_STEPS = 87

# Global interpreter variables
interpreter = None
input_details = None
output_details = None

# --- LIFECYCLE MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global interpreter, input_details, output_details
    try:
        print(f"Loading TFLite Model: {MODEL_PATH}...")
        
        # Load TFLite Interpreter (Super lightweight, optimized for CPU)
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # Get input and output details once to save time per request
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✅ Fast-Brain (TFLite) Loaded Successfully!")
        print(f"   Input Shape: {input_details[0]['shape']}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load model. {e}")
    yield
    print("Shutting down...")

# --- INIT APP ---
app = FastAPI(title="DeepVoice Guard API", lifespan=lifespan)

# --- SECURITY: CORS MIDDLEWARE (The Bridge to Frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Dev mode: safe for hackathons)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, OPTIONS)
    allow_headers=["*"],  # Allows all headers
)

# --- CORE LOGIC (DEBUG VERSION) ---
def preprocess_audio(file_bytes: io.BytesIO):
    try:
        # 1. Load Audio
        print("🔍 Debug: Loading audio...")
        audio, sr = librosa.load(file_bytes, sr=SAMPLE_RATE, duration=DURATION)
        
        # DEBUG: Check if audio is empty
        print(f"📊 Audio Stats - Shape: {audio.shape}, Max: {np.max(audio)}, Min: {np.min(audio)}")
        if np.max(audio) == 0:
            print("⚠️ WARNING: Audio Signal is SILENCE!")

        # 2. Pad/Crop
        target_len = int(DURATION * SAMPLE_RATE)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        # 3. Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # DEBUG: Check Spectrogram
        print(f"📊 Spec Stats - Mean: {np.mean(mel_spec_db)}, Max: {np.max(mel_spec_db)}")

        # 4. Fix Width
        current_width = mel_spec_db.shape[1]
        if current_width < MAX_TIME_STEPS:
            padding = MAX_TIME_STEPS - current_width
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding)))
        else:
            mel_spec_db = mel_spec_db[:, :MAX_TIME_STEPS]

        # 5. Normalize
        mel_spec_db = (mel_spec_db + 80) / 80

        # 6. Reshape
        input_data = mel_spec_db.reshape(1, N_MELS, MAX_TIME_STEPS)
        return input_data.astype(np.float32)

    except Exception as e:
        print(f"❌ Preprocessing Error: {e}")
        raise ValueError("Failed to process audio file")

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not interpreter:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        print(f"🚀 Received File: {file.filename}")
        contents = await file.read()
        audio_stream = io.BytesIO(contents)
        
        # Run Preprocessing
        input_tensor = preprocess_audio(audio_stream)

        # Run Inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # DEBUG: Raw Output
        raw_pred = output_data[0][0]
        print(f"🧠 Model Raw Output: {raw_pred}")

        confidence = float(raw_pred)
        is_fake = confidence > 0.5 
        
        return {
            "filename": file.filename,
            "label": "FAKE" if is_fake else "REAL",
            "confidence": round(confidence, 4),
            "is_fake": is_fake,
            "engine": "TFLite Edge"
        }

    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
