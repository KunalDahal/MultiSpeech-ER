import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

class VideoEmbedder:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def embed(self, video_path):
        cap = cv2.VideoCapture(video_path)
        embeddings = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.transform(frame).unsqueeze(0)

            with torch.no_grad():
                emb = self.model(frame)

            embeddings.append(emb)

        cap.release()

        return torch.mean(torch.stack(embeddings), dim=0)