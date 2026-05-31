import io
import cv2
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NusantaraLens API", description="API Klasifikasi Gambar Budaya Indonesia")

# 1. SETUP CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. BATASAN UKURAN FILE
MAX_FILE_SIZE = 5 * 1024 * 1024

# Load Model
MODEL_PATH = "model_nusantara_lens.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

LABEL_MAP = {0: "Kuliner", 1: "Lagu_Daerah", 2: "Pahlawan", 3: "Tarian"}

# 3. LOAD DATA JSON
try:
    with open("Data deksripsi budaya.json", "r", encoding="utf-8") as f:
        DATA_BUDAYA = json.load(f)
    print("File Data deksripsi budaya.json berhasil dimuat!")
except Exception as e:
    print("File Data deksripsi budaya.json tidak ditemukan!")
    DATA_BUDAYA = []

def preprocess_image_consistent(img):
    h, w, _ = img.shape
    min_dim = min(h, w)
    start_x = w // 2 - min_dim // 2
    start_y = h // 2 - min_dim // 2
    cropped_img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    img_resized = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
    img_array = img_resized.astype("float32")
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return np.expand_dims(img_array, axis=0)

@app.get("/")
def read_root():
    return {"message": "Server NusantaraLens API aktif dan berjalan."}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model AI belum siap di server.")

    # Proteksi 1: Cek Format File
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPEG/PNG).")

    # Proteksi 2: Baca isi gambar sekaligus cek ukuran file-nya
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Ukuran gambar terlalu besar! Maksimal 5MB.")

    try:
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Gambar rusak atau format tidak didukung.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_batch = preprocess_image_consistent(img)

        predictions = model.predict(img_batch)
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])

        kategori_hasil = LABEL_MAP.get(predicted_class_index, "Tidak diketahui")
        rekomendasi_budaya = [item for item in DATA_BUDAYA if item.get("Kategori") == kategori_hasil]

        return JSONResponse(content={
            "status": "success",
            "kategori_tebakan_ai": kategori_hasil,
            "confidence_percentage": round(confidence * 100, 2),
            "jumlah_data_ditemukan": len(rekomendasi_budaya),
            "daftar_rekomendasi": rekomendasi_budaya
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pemrosesan: {str(e)}")
