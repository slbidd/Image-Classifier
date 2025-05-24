import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from model.efficientnet_b0 import build_model


def main():
    print("[Info] 确保输出目录存在...")
    os.makedirs('../outputs', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] 使用设备: {device}")

    print("[Info] 正在构建图像预处理器...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print("[Info] 正在加载训练集和验证集...")
    train_ds = datasets.ImageFolder('../data/train', transform)
    val_ds = datasets.ImageFolder('../data/val', transform)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)
    print(f"[Info] 训练样本数: {len(train_ds)}, 验证样本数: {len(val_ds)}")

    print("[Info] 正在构建模型...")
    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("[Info] 开始训练...")
    best_acc, history = 0.0, {"train_loss": [], "val_acc": []}
    for epoch in range(5):
        print(f"\n[Epoch {epoch + 1}] 训练中...")
        model.train()
        total_loss = 0.0
        for i, (inputs, labels) in enumerate(train_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 10 == 0 or (i + 1) == len(train_dl):
                print(f"  [Batch {i + 1}/{len(train_dl)}] 当前 Loss: {loss.item():.4f}")

        print("[Info] 验证中...")
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"[Result] Epoch {epoch + 1} 完成 - Loss: {total_loss:.4f}, Val Acc: {acc:.4f}")
        history['train_loss'].append(total_loss)
        history['val_acc'].append(acc)

        if acc > best_acc:
            best_acc = acc
            print("[Info] 准确率提升，保存新模型到 ../outputs/model_best.pth")
            torch.save(model.state_dict(), '../outputs/model_best.pth')

    print("[Info] 保存训练指标到 ../outputs/metrics.json")
    with open('../outputs/metrics.json', 'w') as f:
        json.dump(history, f)

    print("[Info] 训练完成！")


if __name__ == '__main__':
    main()
