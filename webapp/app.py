import os
import torch
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from model import build_model
from settings import device, model_path
from data import val_test_transforms
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 清理上传文件夹
import glob

for file_path in glob.glob(os.path.join(UPLOAD_FOLDER, '*')):
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"删除失败: {file_path} - {e}")

model = build_model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

from data import datasets
class_names = datasets["test"].classes

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = val_test_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        predicted_class = class_names[pred.item()]
        confidence = conf.item()
        return predicted_class, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []

    if request.method == 'POST':
        files = request.files.getlist('file')
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                label, prob = predict_image(filepath)
                results.append({
                    "filename": filename,
                    "label": label,
                    "confidence": f"{prob * 100:.2f}%",
                    "url": f"/static/uploads/{filename}"
                })

    return render_template("index.html", results=results)

if __name__ == '__main__':
    app.run(debug=True)
