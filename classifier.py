import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO


class ImageClassifier:
    def __init__(self, model_path="best_model.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["Cat", "Dog"]
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(BytesIO(image))
        image = preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted_class = torch.max(probabilities, 1)

        predicted_class = predicted_class.item()
        confidence = confidences.item()

        return self.classes[predicted_class], confidence
