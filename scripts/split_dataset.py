import os, shutil, random


def split_dataset(input_dir, train_dir, val_dir, split_ratio=0.8):
    random.seed(42)
    class_names = os.listdir(input_dir)

    for cls in class_names:
        img_paths = [os.path.join(input_dir, cls, f) for f in os.listdir(os.path.join(input_dir, cls))]
        random.shuffle(img_paths)
        split_point = int(len(img_paths) * split_ratio)

        for i, path in enumerate(img_paths):
            dest = train_dir if i < split_point else val_dir
            os.makedirs(os.path.join(dest, cls), exist_ok=True)
            shutil.copy(path, os.path.join(dest, cls, os.path.basename(path)))


if __name__ == '__main__':
    split_dataset(
        input_dir='../data/raw_images',
        train_dir='../data/train',
        val_dir='../data/val'
    )
