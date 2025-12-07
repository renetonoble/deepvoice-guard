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

# --- CORE LOGIC ---
def preprocess_audio(file_bytes: io.BytesIO):
    """
    Converts raw audio bytes -> Mel Spectrogram -> DB Scale -> Normalized -> Reshaped
    Must replicate training logic exactly.
    """
    try:
        # 1. Load Audio
        # librosa.load resamples the audio to 22050Hz automatically
        audio, _ = librosa.load(file_bytes, sr=SAMPLE_RATE, duration=DURATION)
        
        # 2. Pad/Crop to exactly 2.0s
        target_len = int(DURATION * SAMPLE_RATE)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        # 3. Mel Spectrogram (The 'Image' of the sound)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # 4. Fix Width (Time Steps)
        # STFT windows can vary slightly; force strict shape for the CNN
        current_width = mel_spec_db.shape[1]
        if current_width < MAX_TIME_STEPS:
            padding = MAX_TIME_STEPS - current_width
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding)))
        else:
            mel_spec_db = mel_spec_db[:, :MAX_TIME_STEPS]

        # 5. Normalize (Physics of Neural Networks)
        # Shift values from [-80, 0] dB to [0, 1] range for stability
        mel_spec_db = (mel_spec_db + 80) / 80

        # 6. Reshape for TFLite Input
        # Model Expects: (Batch, Freq, Time) -> (1, 128, 87)
        input_data = mel_spec_db.reshape(1, N_MELS, MAX_TIME_STEPS)
        
        # TFLite requires float32 specifically
        return input_data.astype(np.float32)

    except Exception as e:
        print(f"Preprocessing Error: {e}")
        raise ValueError("Failed to process audio file")

@app.get("/")
async def root():
    return {"status": "DeepVoice Guard (TFLite) is Online"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not interpreter:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Preprocess
        contents = await file.read()
        audio_stream = io.BytesIO(contents)
        input_tensor = preprocess_audio(audio_stream)

        # 2. Inference (TFLite Style)
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        # Run the model
        interpreter.invoke()
        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 3. Result Interpretation
        confidence = float(output_data[0][0])
        # If confidence > 0.5, the model thinks it belongs to Class 1 (Fake)
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
