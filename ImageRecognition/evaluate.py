import os
import cv2
import torch
import matplotlib.pyplot as plt

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "..", "sessions", "best_model.pth")
image_folder = os.path.join(BASE_DIR, "test_images")

print("Model path:", model_path)
print("Image folder:", image_folder)


num_classes = 2  # background + your object class

model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

print("Model loaded successfully!")



for file in os.listdir(image_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):

        image_path = os.path.join(image_folder, file)

        img = cv2.imread(image_path)

        if img is None:
            print(f"Could not read image: {file}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = torch.tensor(img_rgb / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)[0]

        boxes = output["boxes"]
        scores = output["scores"]

        for box, score in zip(boxes, scores):
            if score >= 0.5:
                x1, y1, x2, y2 = box.int().tolist()

                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

                cv2.putText(
                    img_rgb,
                    f"{score:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

        plt.figure(figsize=(8, 6))
        plt.imshow(img_rgb)
        plt.title(file)
        plt.axis("off")
        plt.show()