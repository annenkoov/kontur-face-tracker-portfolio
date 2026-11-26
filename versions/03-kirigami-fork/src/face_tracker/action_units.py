"""Локальная модель Py-Feat для оценки активности FACS Action Units."""

from __future__ import annotations

import math

import numpy as np


# Py-Feat v2 возвращает модельные оценки активности, а не экспертные оценки
# интенсивности FACS по шкале 0–5. Храним AU отдельным набором и не подмешиваем
# их в индекс напряжения без самостоятельной проверки на размеченных данных.
AU_LABELS_RU = {
    "AU01": "Подъём внутренних бровей",
    "AU02": "Подъём внешних бровей",
    "AU04": "Опускание бровей",
    "AU05": "Раскрытие верхних век",
    "AU06": "Подъём щёк",
    "AU07": "Напряжение век",
    "AU09": "Смещение носа",
    "AU10": "Подъём верхней губы",
    "AU11": "Глубина носогубной складки",
    "AU12": "Подтягивание уголков губ",
    "AU14": "Ямочки / подтягивание щёк",
    "AU15": "Опускание уголков губ",
    "AU17": "Подъём подбородка",
    "AU20": "Растяжение губ",
    "AU23": "Сжатие губ",
    "AU24": "Смыкание губ",
    "AU25": "Размыкание губ",
    "AU26": "Опускание челюсти",
    "AU28": "Втягивание губ",
    "AU43": "Закрывание глаз",
}


class PyFeatActionUnitAnalyzer:
    """Инициализирует Py-Feat один раз и считает AU на текущем face crop."""

    def __init__(self) -> None:
        # Импорт намеренно отложен: основной трекер продолжает работать даже
        # если необязательный AU-пакет или его веса недоступны в этой среде.
        import torch
        from feat import Detectorv2

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Отключаем identity-head: идентификация человека приложению не нужна.
        self._detector = Detectorv2(device=device, identity_model=None)
        self.device = device

    def analyze(self, face_rgb: np.ndarray) -> dict[str, float] | None:
        """Вернуть модельные оценки 0–1, не сохраняя изображение или crop."""
        if face_rgb.ndim != 3 or face_rgb.shape[2] != 3 or face_rgb.size == 0:
            return None
        # Изолированный crop уже получен MediaPipe; размер уменьшаем только для
        # скорости, сохраняя пропорции и детали, нужные для оценки AU.
        height, width = face_rgb.shape[:2]
        scale = min(1.0, 512.0 / max(height, width))
        image = face_rgb
        if scale < 1.0:
            import cv2

            image = cv2.resize(face_rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        # Передаём кадр как RGB-тензор NCHW: ImageDataset принимает пути к
        # файлам, а TensorDataset обрабатывает кадры прямо из памяти, без
        # временного сохранения личных изображений на диск.
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        result = self._detector.detect(tensor, data_type="tensor", progress_bar=False,
                                       num_workers=0, batch_size=1)
        if result is None or result.empty:
            return None
        row = result.iloc[0]
        values: dict[str, float] = {}
        for key in AU_LABELS_RU:
            if key not in result.columns:
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                # Контракт модели и UI — относительная активность в диапазоне
                # 0–1; ограничиваем выбросы на случай другой версии чекпойнта.
                values[key] = min(1.0, max(0.0, value))
        return values or None

