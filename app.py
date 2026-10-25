import io
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms
from model import SimpleCNN

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Load the trained weights once, when the server starts
model = SimpleCNN()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()   # turns dropout off

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),   # the model only knows 32x32 images
    transforms.ToTensor(),
])

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")  # force 3 channels
    x = preprocess(image).unsqueeze(0)          # [3,32,32] -> [1,3,32,32], a batch of 1
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]   # scores -> probabilities
    top = probs.argmax().item()
    return {"prediction": CLASSES[top], "confidence": round(probs[top].item(), 3)}