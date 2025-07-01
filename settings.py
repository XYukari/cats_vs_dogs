import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model.pth"

num_epochs = 20
patience = 3
learning_rate = 0.001
batch_size = 128
