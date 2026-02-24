from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import shutil
import os
from PIL import Image, ImageOps
import pillow_heif
import insightface

pillow_heif.register_heif_opener()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=-1)

faces_folder = "faces"
os.makedirs(faces_folder, exist_ok=True)

known_embeddings = []
known_names = []

def normalize(v):
    return v / np.linalg.norm(v)

def load_faces():
    global known_embeddings, known_names
    known_embeddings = []
    known_names = []

    for filename in os.listdir(faces_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(faces_folder, filename)
            img = Image.open(path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            img_np = np.array(img)

            faces = model.get(img_np)
            if faces:
                emb = normalize(faces[0].embedding)
                name = filename.split("_")[0]
                known_embeddings.append(emb)
                known_names.append(name)

load_faces()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<link rel="apple-touch-icon" href="/static/icon.png">
<title>Face Recognition</title>

<style>
body {
    margin:0;
    font-family:-apple-system,BlinkMacSystemFont;
    background: linear-gradient(135deg,#1f2937,#111827);
    color:white;
    display:flex;
    justify-content:center;
    align-items:center;
    height:100vh;
}
.container {
    width:95%;
    max-width:400px;
    background:rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-radius:20px;
    padding:25px;
    box-shadow:0 20px 50px rgba(0,0,0,0.5);
}
.file-label {
    display:block;
    background:#2563eb;
    padding:14px;
    text-align:center;
    border-radius:12px;
    cursor:pointer;
    font-weight:600;
    margin-bottom:15px;
}
input[type="file"] { display:none; }
.preview { width:100%; border-radius:15px; margin-bottom:15px; display:none; }
button {
    width:100%;
    padding:14px;
    border:none;
    border-radius:12px;
    background:#10b981;
    color:white;
    font-size:16px;
    font-weight:600;
}
.result { margin-top:20px; text-align:center; font-size:18px; font-weight:bold; }
.loader { margin-top:15px; text-align:center; display:none; }
</style>
</head>

<body>
<div class="container">
    <h2>Face Recognition</h2>

    <label class="file-label">
        Выбрать фото
        <input type="file" id="fileInput" accept="image/*">
    </label>

    <img id="preview" class="preview">
    <button onclick="upload()">Распознать</button>

    <div class="loader" id="loader">Обработка...</div>
    <div class="result" id="result"></div>
</div>

<script>
const input = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const loader = document.getElementById("loader");
const result = document.getElementById("result");

let selectedFile = null;

input.onchange = function(e){
    selectedFile = e.target.files[0];
    preview.src = URL.createObjectURL(selectedFile);
    preview.style.display = "block";
    result.innerText = "";
};

async function upload(){
    if(!selectedFile) return alert("Выберите фото");

    loader.style.display = "block";
    result.innerText = "";

    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/recognize",{
        method:"POST",
        body:formData
    });

    const data = await response.json();
    loader.style.display = "none";

    if(data.person === "Unknown"){
        result.innerHTML = "❌ Неизвестно";
        result.style.color = "#f87171";
    }
    else{
        result.innerHTML = "✅ " + data.person;
        result.style.color = "#34d399";
    }
}
</script>
</body>
</html>
"""

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    temp_path = "temp_image"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = Image.open(temp_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img_np = np.array(img)

    faces = model.get(img_np)

    os.remove(temp_path)

    if not faces:
        return {"person": "No face detected"}

    unknown_embedding = normalize(faces[0].embedding)

    distances = [np.linalg.norm(unknown_embedding - emb) for emb in known_embeddings]
    best_index = np.argmin(distances)

    if distances[best_index] < 0.8:
        return {"person": known_names[best_index]}

    return {"person": "Unknown"}
