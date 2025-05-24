import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model.efficientnet_b0 import build_model

label_map = {0: "class1", 1: "class2"}


# 提前加载模型
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load('../outputs/model_best.pth', map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device


import time

def predict(model, device, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    try:
        img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print(f"无法打开图片：{e}")
        return

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = int(outputs.argmax(1).item())

    end_time = time.time()

    print(f"原始输出 logits：{outputs.cpu().numpy()[0]}")
    print(f"Softmax 概率：{probs}")
    print(f"预测结果：{label_map[pred_class]}")
    print(f"推理用时：{(end_time - start_time) * 1000:.2f} ms")

    if device.type == "cuda":
        used_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"显存占用：{used_memory:.2f} MB")

    print("-" * 50)


if __name__ == '__main__':
    model, device = load_model()
    while True:
        path = input("请输入图片路径（输入 q 退出）：").strip()
        if path.lower() == 'q':
            break
        predict(model, device, path)
