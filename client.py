import io
import requests
from torchvision import datasets

test = datasets.CIFAR10(root="data", train=False)   # real test images, no tensors

for i in range(5):
    img, label = test[i]
    buf = io.BytesIO()
    img.save(buf, format="PNG")      # turn the image into file bytes
    buf.seek(0)
    r = requests.post("http://127.0.0.1:8000/predict",
                      files={"file": ("test.png", buf, "image/png")})
    print(f"actual: {test.classes[label]:10} | api says: {r.json()}")