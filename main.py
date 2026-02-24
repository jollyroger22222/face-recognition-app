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

# ЛЁГКАЯ модель для Render Free
model = insightface.app.FaceAnalysis(name='buffalo_s')
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

    print("Loading faces...")

    for filename in os.listdir(faces_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(faces_folder, filename)

            img = Image.open(path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            img_np = np.array(img)

            faces = model.get(img_np)

            if len(faces) > 0:
                emb = normalize(faces[0].embedding)
                name = filename.split("_")[0]
                known_embeddings.append(emb)
                known_names.append(name)

                print("Loaded:", filename)

    print("Total loaded:", len(known_embeddings))

load_faces()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<link rel="apple-touch-icon" href="/static/icon.png">
<title>Face Recognition</title>
</head>
<body style="background:#111;color:white;font-family:sans-serif;text-align:center;padding:30px">
<h2>Face Recognition</h2>
<input type="file" id="fileInput" accept="image/*">
<br><br>
<button onclick="upload()">Распознать</button>
<p id="result"></p>

<script>
let selectedFile=null;
document.getElementById("fileInput").onchange=function(e){
    selectedFile=e.target.files[0];
};
async function upload(){
    if(!selectedFile){alert("Выберите фото");return;}
    const formData=new FormData();
    formData.append("file",selectedFile);
    const response=await fetch("/recognize",{method:"POST",body:formData});
    const data=await response.json();
    document.getElementById("result").innerText=data.person;
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
