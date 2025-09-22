import torch
import torchvision.transforms as transforms
from PIL import Image

def classify_single_png(png_path, model, transform, threshold, device):
    """
    Классифицирует один PNG-файл с использованием переданной модели.

    Аргументы:
        png_path (str): путь к PNG-файлу
        model: загруженная PyTorch модель (должна быть в .eval() режиме)
        transform: torchvision.transforms.Compose — трансформ для входного изображения
        threshold (float): порог для бинарной классификации
        device: torch.device — устройство для вычислений

    Возвращает:
        tuple: (prediction: str, probability: float)
        где prediction — "Норма" или "Патология"
    """
    try:
        # Открываем изображение в grayscale
        image = Image.open(png_path).convert('L')

        # Применяем трансформ
        input_tensor = transform(image).unsqueeze(0).to(device)  # добавляем batch-размерность

        # Предсказание
        with torch.no_grad():
            output = model(input_tensor)
            # Предполагаем, что выход — logits для 2 классов
            prob = torch.sigmoid(output[0, 0]).item()

        # Применяем порог
        prediction = "Патология" if prob >= threshold else "Норма"

        return prediction, prob

    except Exception as e:
        raise RuntimeError(f"Ошибка при классификации {png_path}: {str(e)}")
