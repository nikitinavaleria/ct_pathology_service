import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def create_classifier_heatmap(activation, gradient, lung_mask, output_path):
    """
    Создает и сохраняет тепловую карту на основе Grad-CAM для классификатора.
    activation: Grad-CAM карта (H, W)
    gradient: Средние градиенты (для совместимости)
    lung_mask: Маска легких
    output_path: Путь для сохранения тепловой карты
    """
    output_path = Path(output_path)
    debug_dir = output_path.parent
    filename_stem = output_path.stem

    log_file = debug_dir / f"{filename_stem}_heatmap.log"
    with open(log_file, "w") as f:
        print(f"  Activation stats: min={activation.min():.4f}, max={activation.max():.4f}, mean={activation.mean():.4f}", file=f)
        print(f"  Gradient stats: min={gradient.min():.4f}, max={gradient.max():.4f}, mean={gradient.mean():.4f}", file=f)
        print(f"  Lung mask stats: sum={lung_mask.sum():.2f}, max={lung_mask.max():.4f}", file=f)

    # Применение маски легких
    if lung_mask.max() > 0 and lung_mask.sum() > 100:
        activation = activation * lung_mask
        activation -= activation.min()
        if activation.max() > 0:
            activation /= activation.max()
        activation = (activation * 255).astype(np.uint8)
    else:
        print("  ⚠️ Предупреждение: Маска легких пустая или слишком маленькая. Используем cam без маски.", file=open(log_file, "a"))

    # Применение цветовой карты
    heatmap_color = cv2.applyColorMap(activation, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Сохранение
    Image.fromarray(heatmap_color).save(output_path)
    with open(log_file, "a") as f:
        print(f"  ✅ Тепловая карта классификатора сохранена: {output_path}", file=f)

    # Сохранение отладочных данных
    Image.fromarray(activation).save(debug_dir / f"{filename_stem}.cam.png")
    with open(log_file, "a") as f:
        print(f"  ✅ CAM сохранен: {debug_dir / f'{filename_stem}.cam.png'}", file=f)

    Image.fromarray((lung_mask * 255).astype(np.uint8)).save(debug_dir / f"{filename_stem}.mask.png")
    with open(log_file, "a") as f:
        print(f"  ✅ Маска легких сохранена: {debug_dir / f'{filename_stem}.mask.png'}", file=f)

    # Отладочная визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(activation, cmap='jet')
    axes[0].set_title('CAM')
    axes[0].axis('off')

    axes[1].imshow(lung_mask, cmap='gray')
    axes[1].set_title('Lung Mask')
    axes[1].axis('off')

    axes[2].imshow(heatmap_color)
    axes[2].set_title('Heatmap')
    axes[2].axis('off')

    plt.savefig(debug_dir / f"{filename_stem}_debug_plot.png")
    plt.close()
    with open(log_file, "a") as f:
        print(f"  ✅ Отладочная визуализация сохранена: {debug_dir / f'{filename_stem}_debug_plot.png'}", file=f)