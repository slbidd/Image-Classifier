from torchvision import datasets, transforms

# 用于确认 ImageFolder 自动分配的类别索引
def check_class_indices(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    print("class_to_idx:", dataset.class_to_idx)

 # 训练集路径
if __name__ == '__main__':
    check_class_indices('../data/train')