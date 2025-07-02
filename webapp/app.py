import glob
import os

from PIL import Image, UnidentifiedImageError
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

from classifier import ImageClassifier

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

for file_path in glob.glob(os.path.join(UPLOAD_FOLDER, '*')):
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"删除失败: {file_path} - {e}")

classifier = ImageClassifier(model_path="best_model.pth")

def resize_img(image, size=(224, 224)):
    from PIL import Image
    return image.resize(size, Image.Resampling.LANCZOS)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("file")
        results = []

        for file in files:
            if file:
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                    image = Image.open(file)
                    resized_img = resize_img(image)
                    resized_img.save(file_path)

                    with open(file_path, "rb") as img_file:
                        image_data = img_file.read()

                    label, confidence = classifier.predict(image_data)

                    results.append({
                        'filename': filename,
                        'url': url_for('static', filename='uploads/' + filename),
                        'label': label,
                        'confidence': f"{confidence * 100:.2f}%"
                    })

                except UnidentifiedImageError:
                    results.append({
                        'filename': filename,
                        'error': "无法识别的图像文件"
                    })

                except Exception as e:
                    results.append({
                        'filename': filename,
                        'error': f"处理失败: {str(e)}"
                    })

        return render_template("index.html", results=results)

    return render_template("index.html")
