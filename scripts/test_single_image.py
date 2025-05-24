import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model.efficientnet_b0 import build_model

label_map = {0: "class1", 1: "class2"}


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load('../outputs/model_best.pth', map_location=device, weights_only=True))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)  # [1, 2]
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]  # 概率 [p1, p2]
        pred_class = int(outputs.argmax(1).item())

    print(f"原始输出 logits：{outputs.cpu().numpy()[0]}")
    print(f"Softmax 概率：{probs}")
    print(f"预测结果：{label_map[pred_class]}")


if __name__ == '__main__':
    predict(input("请输入图片路径"))