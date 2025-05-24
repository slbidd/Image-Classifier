import json
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=12)

with open("../outputs/metrics.json") as f:
    history = json.load(f)

plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.title("训练损失", fontproperties=font)
plt.xlabel("Epoch", fontproperties=font)
plt.ylabel("Loss", fontproperties=font)
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history["val_acc"], label="Validation Acc")
plt.title("验证准确率", fontproperties=font)
plt.xlabel("Epoch", fontproperties=font)
plt.ylabel("Accuracy", fontproperties=font)
plt.grid()

plt.tight_layout()
plt.savefig("../outputs/training_plot.png")
plt.show()
