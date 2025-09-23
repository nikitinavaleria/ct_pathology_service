# import torch
# import torchvision.transforms as transforms
# from PIL import Image
#
# def classify_single_png(png_path, model, transform, threshold, device):
#     """
#     Классифицирует один PNG-файл с использованием переданной модели.
#
#     Аргументы:
#         png_path (str): путь к PNG-файлу
#         model: загруженная PyTorch модель (должна быть в .eval() режиме)
#         transform: torchvision.transforms.Compose — трансформ для входного изображения
#         threshold (float): порог для бинарной классификации
#         device: torch.device — устройство для вычислений
#
#     Возвращает:
#         tuple: (prediction: str, probability: float)
#         где prediction — "Норма" или "Патология"
#     """
#     try:
#
#         # Открываем изображение в grayscale
#         image = Image.open(png_path).convert('L')
#
#
#         # Применяем трансформ
#         input_tensor = transform(image).unsqueeze(0).to(device)  # добавляем batch-размерность
#         print('Эщкере3')
#         # Предсказание
#         with torch.no_grad():
#             print('Эщкере4')
#             output = model(input_tensor)
#             print('Эщкере5')
#             # Предполагаем, что выход — logits для 2 классов
#             prob = torch.sigmoid(output[0, 0]).item()
#
#         # Применяем порог
#         prediction = "Патология" if prob >= threshold else "Норма"
#
#         return prediction, prob
#
#     except Exception as e:
#         raise RuntimeError(f"Ошибка при классификации {png_path}: {str(e)}")


# backend/app/ml/classifier.py
import time, os
import numpy as np
import cv2
import torch
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        maxv = int(img.max()) or 1
        img = (img.astype(np.float32) * (255.0 / maxv)).clip(0, 255).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img

def classify_single_png(png_path, model, transform, threshold, device):
    t0 = time.perf_counter()
    print(f"[DBG] start classify_single_png: {png_path}")

    # читаем PNG (может быть 8/16 бит, 1/3 канал)
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cv2.imread returned None for {png_path}")
    print(f"[DBG] read: shape={img.shape}, dtype={img.dtype}, t={time.perf_counter()-t0:.4f}s")

    img = _to_uint8(img)

    # -> строго в GRAY (1 канал)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 512x512
    # если уже 2D — оставляем как есть

    print(f"[DBG] to GRAY: shape={img.shape}, t={time.perf_counter()-t0:.4f}s")

    # resize к 224x224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # HxW uint8 -> 1xHxW float32 [0..1] -> нормализация (один канал)
    x = img.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5  # при необходимости подстрой под твои статистики обучения
    x = x[None, :, :]    # 1,H,W
    xt = torch.from_numpy(x).unsqueeze(0).contiguous()  # 1,1,224,224

    print(f"[DBG] tensor ready: {tuple(xt.shape)} dtype={xt.dtype} t={time.perf_counter()-t0:.4f}s")

    # девайс и форвард
    try:
        model.to(device)
    except Exception as e:
        print(f"[DBG] WARNING: model.to({device}) failed: {e}")
    xt = xt.to(device, non_blocking=False)
    print(f"[DBG] devices: input={xt.device} model_params={next(model.parameters()).device}")

    model.eval()
    with torch.no_grad():
        t1 = time.perf_counter()
        logits = model.forward(xt)
        print(f"[DBG] forward ok in {time.perf_counter()-t1:.4f}s; logits.shape={tuple(logits.shape)}")

    # бинарная вероятность
    if logits.shape[1] == 1:
        prob = torch.sigmoid(logits[0, 0]).item()
    else:
        p = F.softmax(logits, dim=1)[0]
        prob = float(p[1].item() if p.shape[0] > 1 else p[0].item())

    pred = "Патология" if prob >= float(threshold) else "Норма"
    print(f"[DBG] done: pred={pred} prob={prob:.4f} total={time.perf_counter()-t0:.4f}s")
    return pred, prob
