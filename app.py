# app.py

from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as T
from torch import nn
from PIL import Image
import os
import uuid

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "give_way",
    "no_entry",
    "priority_road",
    "stop",
    "unknown"
]

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

# -----------------------------
# IMAGE TRANSFORMS
# -----------------------------

test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean_nums, std_nums)
])

# -----------------------------
# LOAD MODEL
# -----------------------------

def create_model(n_classes):
    model = models.resnet34(pretrained=False)

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)

    return model

model = create_model(len(class_names))

model.load_state_dict(
    torch.load("traffic_sign_model.pth", map_location=device)
)

model = model.to(device)
model.eval()

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = test_transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = F.softmax(outputs, dim=1)

        confidence, pred = torch.max(probabilities, 1)

    predicted_class = class_names[pred.item()]
    confidence_score = confidence.item() * 100

    return predicted_class, round(confidence_score, 2)

# -----------------------------
# ROUTES
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    image_file = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template("index.html")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html")

        filename = str(uuid.uuid4()) + ".jpg"

        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        prediction, confidence = predict_image(filepath)

        image_file = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_file=image_file
    )

# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)