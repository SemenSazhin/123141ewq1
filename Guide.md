```markdown
# 🐕 Инструкция по использованию Google Colab блокнота PoopDetector Pro

## 📑 Оглавление

1. [Первоначальная настройка](#1-первоначальная-настройка)
2. [Структура блокнота и порядок выполнения](#2-структура-блокнота-и-порядок-выполнения)
3. [Описание классов и их методов](#3-описание-классов-и-их-методов)
4. [Подготовка данных для обучения](#4-подготовка-данных-для-обучения)
5. [Обучение LSTM модели](#5-обучение-lstm-модели)
6. [Использование модели для инференса](#6-использование-модели-для-инференса)
7. [Экспорт и загрузка модели](#7-экспорт-и-загрузка-модели)
8. [Решение типичных проблем](#8-решение-типичных-проблем)

---

## 1. Первоначальная настройка

### 1.1 Открытие блокнота в Colab

1. Перейдите по ссылке на блокнот или загрузите `.ipynb` файл в Google Drive
2. Откройте файл в Google Colab
3. В меню выберите: **Runtime → Change runtime type**
4. Установите **Hardware accelerator = GPU (T4 рекомендуется)**
5. Нажмите **Save**

### 1.2 Подключение Google Drive

Перед началом работы необходимо подключить Google Drive для доступа к моделям и данным:

```python
# Ячейка 2: Импорт библиотек и настройка (частично)
from google.colab import drive
drive.mount('/content/drive')
```

**Что нужно сделать:**
1. Выполнить ячейку
2. Перейти по появившейся ссылке
3. Скопировать код авторизации
4. Вставить код в поле ввода и нажать Enter

### 1.3 Проверка путей к моделям

В ячейке **Конфигурация** проверьте пути к файлам моделей:

```python
class Config:
    # Замените на ваши актуальные пути
    DOG_DETECT_MODEL_PATH = "/content/drive/MyDrive/.../dog_detect_model.pt"
    DOG_POSE_MODEL_PATH = "/content/drive/MyDrive/.../dog_pose_model.pt"
    LSTM_MODEL_PATH = "/content/drive/MyDrive/.../lstm_model.pth"
```

---

## 2. Структура блокнота и порядок выполнения

Блокнот разделен на **25 ячеек**, которые нужно выполнять последовательно:

### 🔧 Секция 1: Настройка окружения (Ячейки 1-3)

| № | Название | Назначение | Обязательность |
|---|----------|------------|----------------|
| 1 | Установка зависимостей | Установка всех библиотек | ⭐⭐⭐ Обязательно |
| 2 | Импорт библиотек | Импорт модулей, настройка GPU | ⭐⭐⭐ Обязательно |
| 3 | Конфигурация | Определение параметров системы | ⭐⭐⭐ Обязательно |

### 🔬 Секция 2: Определение классов (Ячейки 4-8)

| № | Название | Класс | Назначение |
|---|----------|-------|------------|
| 4 | Процессор позы | `ImprovedPoseProcessor` | Фильтрация и сглаживание ключевых точек |
| 5 | Трекер собак | `RobustDogTracker` | Отслеживание собак с реидентификацией |
| 6 | Экстрактор признаков | `EfficientFeatureExtractor` | Извлечение компактных признаков |
| 7 | LSTM модель | `CompactLSTMPoseClassifier` | Классификация последовательностей |
| 8 | Детектор | `ImprovedDefecationDetector` | Основной класс системы |

### 📦 Секция 3: Инициализация (Ячейки 9-10)

| № | Название | Действие |
|---|----------|----------|
| 9 | Загрузка моделей | Загрузка YOLO моделей |
| 10 | Инициализация детектора | Создание экземпляра детектора |

### 🎬 Секция 4: Инференс (Ячейка 11)

| № | Название | Действие |
|---|----------|----------|
| 11 | Обработка видео | Запуск детекции на видео |

### 💾 Секция 5: Скачивание и тесты (Ячейки 12-15)

| № | Название | Действие |
|---|----------|----------|
| 12 | Скачивание результата | Загрузка обработанного видео |
| 13 | Тест на изображении | Отладка на одном кадре |
| 14 | Бенчмарк | Измерение производительности |
| 15 | Экспорт модели | Сохранение LSTM модели |

### 🎓 Секция 6: Обучение (Ячейки 16-24)

| № | Название | Действие |
|---|----------|----------|
| 16 | Подготовка данных | Классы для подготовки датасета |
| 17 | Обучение LSTM | Класс тренера |
| 18 | Загрузка данных | Извлечение признаков из видео |
| 19 | Запуск обучения | Обучение модели |
| 20 | Оценка модели | Метрики качества |
| 21 | Анализ Attention | Визуализация весов внимания |
| 22 | Экспорт модели | Сохранение для production |
| 23 | Использование | Интеграция в детектор |
| 24 | Дообучение | Fine-tuning на новых данных |

---

## 3. Описание классов и их методов

### 3.1 `Config` - Конфигурация системы

**Где находится:** Ячейка 3

**Назначение:** Централизованное хранение всех параметров системы

**Основные атрибуты:**

```python
config = Config()

# Пороги детекции
config.DETECTION_CONFIDENCE = 0.3  # Мин. уверенность детекции собаки
config.POSE_CONFIDENCE = 0.5       # Мин. уверенность ключевых точек
config.DEFECATION_THRESHOLD = 0.7  # Порог для классификации

# Временные параметры (в секундах)
config.DEFECATION_MIN_DURATION = 2.0   # Длительность позы для подтверждения
config.CLEANING_TIMEOUT = 15.0         # Время на уборку
config.CLEANING_MIN_DURATION = 5.0     # Длительность уборки для подтверждения

# Параметры трекинга
config.TRACKER_MAX_AGE = 45    # Макс. кадров без обновления
config.TRACKER_N_INIT = 3      # Кадров для инициализации трека
```

**Как изменить под свои нужды:**

```python
# Для чувствительной детекции
config.DEFECATION_THRESHOLD = 0.6
config.DETECTION_CONFIDENCE = 0.25

# Для подавления ложных срабатываний
config.DEFECATION_THRESHOLD = 0.85
config.DEFECATION_MIN_DURATION = 3.0
```

---

### 3.2 `ImprovedPoseProcessor` - Процессор позы

**Где находится:** Ячейка 4

**Назначение:** Фильтрация шума, сглаживание и интерполяция ключевых точек

**Основные методы:**

| Метод | Параметры | Возвращает | Описание |
|-------|-----------|------------|----------|
| `filter_keypoints()` | `keypoints`, `confidences` | `ndarray` | Фильтрация и сглаживание точек |
| `get_stable_defecation_point()` | `keypoints` | `tuple(x,y)` | Стабильная точка дефекации |
| `get_keypoint_confidence()` | `keypoints` | `float` | Оценка уверенности в позе |

**Пример использования:**

```python
processor = ImprovedPoseProcessor(
    num_keypoints=20,
    confidence_threshold=0.5,
    smooth_window=5
)

# Фильтрация ключевых точек
filtered_kps = processor.filter_keypoints(raw_keypoints, confidences)

# Получение точки дефекации
defecation_point = processor.get_stable_defecation_point(filtered_kps)
```

**Важные атрибуты:**

```python
# Группы ключевых точек (индексы)
processor.keypoint_groups = {
    'spine': [0, 1, 2, 3, 4, 5],   # Позвоночник
    'hips': [6, 7, 8, 9],          # Таз/бедра
    'tail': [10, 11, 12],          # Хвост, 11 - анус (ключевая!)
}

# Приоритет точек для определения дефекации
processor.defecation_priority = [11, 8, 7, 6, 9]
```

---

### 3.3 `RobustDogTracker` - Трекер собак

**Где находится:** Ячейка 5

**Назначение:** Отслеживание собак с реидентификацией и выбором лучшего трека

**Основные методы:**

| Метод | Параметры | Возвращает | Описание |
|-------|-----------|------------|----------|
| `update()` | `frame`, `detections` | `tracks`, `primary_track` | Обновление треков |
| `get_track_confidence()` | `track` | `float` | Оценка уверенности в треке |
| `_select_primary_track()` | `tracks` | `track` | Выбор основного трека |

**Пример использования:**

```python
tracker = RobustDogTracker(
    max_age=45,      # Дольше помнить трек
    n_init=3,        # Кадров для подтверждения
    max_iou_distance=0.7
)

# На каждом кадре
tracks, primary_track = tracker.update(frame, dog_detections)

if primary_track:
    bbox = primary_track.to_ltrb()  # [x1, y1, x2, y2]
    track_id = primary_track.track_id
    confidence = tracker.get_track_confidence(primary_track)
```

**Настройка под сценарий:**

```python
# Для быстрых движений (больше пропусков)
tracker = RobustDogTracker(max_age=60, max_iou_distance=0.5)

# Для статичных сцен (строже)
tracker = RobustDogTracker(max_age=30, max_iou_distance=0.8)
```

---

### 3.4 `EfficientFeatureExtractor` - Экстрактор признаков

**Где находится:** Ячейка 6

**Назначение:** Извлечение компактных признаков из ключевых точек (сжатие 11400 → ~100)

**Основные методы:**

| Метод | Параметры | Возвращает | Описание |
|-------|-----------|------------|----------|
| `extract_features()` | `keypoints`, `prev_keypoints` | `ndarray` | Извлечение признаков |
| `get_feature_dim()` | - | `int` | Размерность вектора признаков |

**Пример использования:**

```python
extractor = EfficientFeatureExtractor(num_keypoints=20)

# Извлечение признаков
features = extractor.extract_features(current_kps, prev_kps)
delta = features - prev_features

# Размерность
feature_dim = extractor.get_feature_dim()  # ~100
```

**Что входит в признаки:**

```python
# 1. Нормализованные координаты (первые 15 точек) - 30 значений
# 2. Длины ключевых сегментов - ~25 значений
# 3. Ключевые углы - 5 значений
# 4. Скорости точек - 10 значений
# 5. Специфичные признаки дефекации - 3 значения
# 6. Признаки позы - 2 значения
```

---

### 3.5 `CompactLSTMPoseClassifier` - LSTM модель

**Где находится:** Ячейка 7

**Назначение:** Классификация последовательности поз на наличие дефекации

**Архитектура:**
- Энкодер для сжатия признаков (вход → 128 → 64 → 32)
- Двунаправленная LSTM (2 слоя, hidden=128)
- Attention механизм
- Классификатор (128 → 64 → 1)

**Основные методы:**

| Метод | Параметры | Возвращает | Описание |
|-------|-----------|------------|----------|
| `forward()` | `x` (batch, seq, features) | `logits` | Прямой проход |
| `get_attention_weights()` | `x` | `weights` | Веса внимания |

**Пример использования:**

```python
model = CompactLSTMPoseClassifier(
    input_size=feature_dim * 2,  # base + delta
    lstm_hidden=128,
    num_layers=2,
    dropout=0.3
).to(device)

# Инференс
with torch.no_grad():
    logits = model(sequence_tensor)  # [batch, 1]
    prob = torch.sigmoid(logits)     # [batch, 1]
```

---

### 3.6 `ImprovedDefecationDetector` - Основной детектор

**Где находится:** Ячейка 8

**Назначение:** Объединение всех компонентов в единую систему детекции

**Основные методы:**

| Метод | Параметры | Возвращает | Описание |
|-------|-----------|------------|----------|
| `process_frame()` | `frame` | `vis_frame` | Обработка одного кадра |
| `run_video()` | `video_path`, `output_path` | - | Обработка видеофайла |

**Пример использования:**

```python
# Инициализация
detector = ImprovedDefecationDetector(
    dog_detect_model=dog_model,
    pose_model=pose_model,
    lstm_path="model.pth",
    config=config
)

# Обработка видео
detector.run_video(
    video_path="/content/input.mp4",
    output_path="/content/output.mp4"
)
```

**Внутреннее состояние:**

```python
# Статусы детекции
detector.alert_active          # bool - есть подозрение
detector.defecation_confirmed  # bool - дефекация подтверждена
detector.cleaning_detected     # bool - уборка обнаружена
detector.defecation_point_fixed  # tuple - координаты зоны

# Сброс состояния
detector._reset_defecation_state()
```

---

### 3.7 `LSTMTrainer` - Тренер модели

**Где находится:** Ячейка 17

**Назначение:** Обучение LSTM модели с валидацией и early stopping

**Основные методы:**

| Метод | Параметры | Возвращает | Описание |
|-------|-----------|------------|----------|
| `train()` | `train_loader`, `val_loader`, `epochs`, ... | `model` | Обучение модели |
| `plot_training_history()` | - | - | Визуализация обучения |

**Пример использования:**

```python
trainer = LSTMTrainer(model, device, config)

trained_model = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    learning_rate=1e-4,
    patience=15,
    save_path='best_model.pth'
)

# Показать графики
trainer.plot_training_history()
```

---

### 3.8 `DataPreparator` - Подготовка данных

**Где находится:** Ячейка 16

**Назначение:** Извлечение последовательностей признаков из размеченных видео

**Основные методы:**

| Метод | Параметры | Возвращает | Описание |
|-------|-----------|------------|----------|
| `extract_sequences_from_video()` | `video_path`, `label`, `stride` | `list` | Извлечение из одного видео |
| `prepare_dataset()` | `data_dir`, `classes` | `X`, `y` | Подготовка датасета |

**Пример использования:**

```python
preparator = DataPreparator(detector, config)

X, y = preparator.prepare_dataset(
    data_dir="/content/labeled_videos",
    classes=['normal', 'defecation']
)

# X.shape = (n_sequences, 120, feature_dim*2)
# y.shape = (n_sequences,)
```

---

## 4. Подготовка данных для обучения

### 4.1 Структура директорий

Создайте в Google Drive следующую структуру:

```
MyDrive/
└── poopdetector_data/
    ├── labeled_videos/
    │   ├── normal/           # Видео БЕЗ дефекации
    │   │   ├── video1.mp4
    │   │   ├── video2.mp4
    │   │   └── ...
    │   └── defecation/       # Видео С дефекацией
    │       ├── video1.mp4
    │       ├── video2.mp4
    │       └── ...
    ├── models/
    │   ├── dog_detect_model.pt
    │   └── dog_pose_model.pt
    └── output/
```

### 4.2 Требования к видео

| Параметр | Требование |
|----------|------------|
| Формат | `.mp4` (рекомендуется) |
| Длительность | 5-30 секунд |
| Содержание | Собака видна, одна сцена |
| Разметка | Весь файл = один класс |

### 4.3 Запуск подготовки данных

**Шаг 1:** Выполнить ячейки 1-10 для инициализации детектора

**Шаг 2:** В ячейке 18 указать путь к данным:

```python
DATA_DIR = "/content/drive/MyDrive/poopdetector_data/labeled_videos"
```

**Шаг 3:** Выполнить ячейку 18. Процесс займет время (примерно 1-2 минуты на видео).

**Шаг 4:** После выполнения появятся файлы:
- `/content/X_sequences.npy` - признаки
- `/content/y_labels.npy` - метки

### 4.4 Проверка качества данных

```python
import numpy as np

X = np.load('/content/X_sequences.npy')
y = np.load('/content/y_labels.npy')

print(f"Всего последовательностей: {len(X)}")
print(f"Класс 0 (normal): {np.sum(y==0)}")
print(f"Класс 1 (defecation): {np.sum(y==1)}")
print(f"Размерность признаков: {X.shape[2]}")

# Проверка на NaN
print(f"Содержит NaN: {np.isnan(X).any()}")
print(f"Содержит Inf: {np.isinf(X).any()}")
```

---

## 5. Обучение LSTM модели

### 5.1 Подготовка к обучению

**Шаг 1:** Убедиться, что данные подготовлены (есть файлы `X_sequences.npy` и `y_labels.npy`)

**Шаг 2:** В ячейке 19 проверить параметры:

```python
# Размер батча (уменьшить при Out of Memory)
batch_size = 8

# Количество эпох
epochs = 100

# Скорость обучения
learning_rate = 1e-4

# Early stopping (остановка если нет улучшений)
patience = 15
```

### 5.2 Запуск обучения

**Шаг 1:** Выполнить ячейку 19

**Шаг 2:** Наблюдать за прогрессом:

```
Epoch 001/100 | Train Loss: 0.5234 | Train Acc: 0.723 | Val Loss: 0.4892 | Val Acc: 0.781
Epoch 002/100 | Train Loss: 0.4456 | Train Acc: 0.801 | Val Loss: 0.4123 | Val Acc: 0.833
...
✅ Сохранена лучшая модель (val_loss: 0.3456)
```

**Шаг 3:** После завершения модель сохранится в `/content/best_lstm_model.pth`

### 5.3 Анализ результатов обучения

Выполнить ячейку 20 для оценки:

```python
# Загружается лучшая модель
# Выводится:
# - Accuracy
# - AUC-ROC
# - Classification Report
# - Confusion Matrix
# - ROC Curve
```

### 5.4 Визуализация Attention (Ячейка 21)

Показывает, на какие кадры последовательности модель обращает внимание:

```python
# Вывод:
# 🎯 Наиболее важные кадры: [45, 46, 47, 48]
#    Максимальный вес: 0.234 на кадре 46
```

---

## 6. Использование модели для инференса

### 6.1 С уже обученной LSTM

**Шаг 1:** В ячейке 23 указать путь к модели:

```python
production_detector = ImprovedDefecationDetector(
    dog_detect_model=dog_detect_model,
    pose_model=pose_model,
    lstm_path='/content/best_lstm_model.pth',  # Ваша модель
    config=config
)
```

**Шаг 2:** Указать входное и выходное видео:

```python
INPUT_VIDEO = "/content/drive/MyDrive/poopdetector_data/test_video.mp4"
OUTPUT_VIDEO = "/content/drive/MyDrive/poopdetector_data/output/result.mp4"
```

**Шаг 3:** Запустить обработку:

```python
production_detector.run_video(INPUT_VIDEO, OUTPUT_VIDEO)
```

### 6.2 Без LSTM (только эвристики)

Если LSTM модель не обучена, детектор будет использовать только пороговые значения на основе позы:

```python
detector = ImprovedDefecationDetector(
    dog_detect_model=dog_detect_model,
    pose_model=pose_model,
    lstm_path=None,  # Без LSTM
    config=config
)
```

### 6.3 Тестирование на одном изображении (Ячейка 13)

```python
from google.colab.patches import cv2_imshow

# Загрузка изображения
image = cv2.imread("/content/test_image.jpg")

# Обработка
vis_image = detector.process_frame(image)

# Отображение
cv2_imshow(vis_image)
```

### 6.4 Оценка производительности (Ячейка 14)

```python
benchmark_detector(
    video_path="/content/test_video.mp4",
    num_frames=100  # Количество кадров для теста
)

# Вывод:
# 📈 Результаты бенчмарка:
#    Среднее время на кадр: 45.2 мс
#    FPS: 22.1
```

---

## 7. Экспорт и загрузка модели

### 7.1 Экспорт обученной LSTM (Ячейка 22)

```python
export_path = export_trained_model(
    model=trained_model,
    config=detector,
    save_path='/content/defecation_lstm_final.pth'
)

# Создаются файлы:
# - defecation_lstm_final.pth (веса модели)
# - defecation_lstm_final_config.json (конфигурация)
```

### 7.2 Загрузка экспортированной модели

```python
from export import load_exported_model

loaded_model, model_config = load_exported_model(
    '/content/defecation_lstm_final.pth'
)

print(model_config)
# {
#   'input_size': 200,
#   'feature_dim': 100,
#   'window_size': 120,
#   'threshold': 0.7
# }
```

### 7.3 Скачивание модели на компьютер

Выполнить ячейку 12 или 22:

```python
from google.colab import files
files.download('/content/defecation_lstm_final.pth')
files.download('/content/defecation_lstm_final_config.json')
```

### 7.4 Использование модели локально

```python
# Локальный Python скрипт
import torch
from model import CompactLSTMPoseClassifier
import json

# Загрузка конфигурации
with open('defecation_lstm_final_config.json', 'r') as f:
    config = json.load(f)

# Создание модели
model = CompactLSTMPoseClassifier(
    input_size=config['input_size'],
    lstm_hidden=128,
    num_layers=2
)

# Загрузка весов
checkpoint = torch.load('defecation_lstm_final.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 8. Решение типичных проблем

### 8.1 "No such file or directory: dog_detect_model"

**Причина:** Не загружены модели YOLO в Google Drive

**Решение:**
1. Загрузите модели в папку на Google Drive
2. Обновите пути в `Config`:

```python
config.DOG_DETECT_MODEL_PATH = "/content/drive/MyDrive/poopdetector_data/models/dog_detect_model.pt"
config.DOG_POSE_MODEL_PATH = "/content/drive/MyDrive/poopdetector_data/models/dog_pose_model.pt"
```

### 8.2 "CUDA out of memory"

**Причина:** Недостаточно памяти GPU (особенно на T4)

**Решение в ячейке 3:**

```python
# Уменьшить окно последовательности
config.WINDOW_SIZE = 60  # вместо 120

# Уменьшить размер батча (ячейка 19)
batch_size = 4  # вместо 8
```

**Решение в Runtime:**
- Runtime → Disconnect and delete runtime
- Runtime → Change runtime type → Выбрать GPU с большей памятью (если доступно)

### 8.3 "Трекер постоянно переключается между собаками"

**Причина:** Несколько собак в кадре с похожими признаками

**Решение в ячейке 3:**

```python
# Увеличить порог переключения
config.TRACKER_MAX_AGE = 60

# В ячейке 5 изменить параметр:
self.track_switch_threshold = 20  # вместо 10
```

### 8.4 "Модель не находит точку дефекации"

**Причина:** Ключевая точка 11 (анус) не детектируется

**Решение в ячейке 4:**

```python
# Изменить приоритет точек
processor.defecation_priority = [8, 7, 6, 9, 11]  # Таз в приоритете
```

### 8.5 "Очень медленная обработка видео"

**Решение:**

```python
# В ячейке 3:
config.VISUALIZE_ALL_DETECTIONS = False  # Не рисовать все детекции
config.VERBOSE_LOGGING = False           # Отключить логи

# В функции run_video добавить пропуск кадров:
frame_skip = 2
if frame_count % frame_skip != 0:
    continue
```

### 8.6 "Ошибка при сохранении модели: No space left"

**Причина:** Закончилось место на Google Drive

**Решение:**
- Очистить корзину Google Drive
- Сохранять модель локально в `/content/` и скачивать
- Удалить ненужные файлы:

```python
import os
os.remove('/content/X_sequences.npy')
os.remove('/content/y_labels.npy')
```

### 8.7 "Ключевые точки рисуются не там где собака"

**Причина:** Неправильное масштабирование при ресайзе ROI

**Решение в ячейке 8 (метод `_process_dog_track`):**

```python
# Убедиться что scale_factor учитывается правильно
if scale_factor != 1.0:
    kps /= scale_factor  # Эта строка должна быть
```

---

## 📋 Чек-лист для запуска

### Для инференса (готовой моделью)

- [ ] Подключен Google Drive
- [ ] Загружены модели YOLO в Drive
- [ ] Указаны правильные пути в Config
- [ ] Выполнены ячейки 1-3 (установка, импорт, конфиг)
- [ ] Выполнены ячейки 4-8 (определение классов)
- [ ] Выполнена ячейка 9 (загрузка моделей)
- [ ] Выполнена ячейка 10 (инициализация детектора)
- [ ] В ячейке 11 указан путь к видео
- [ ] Выполнена ячейка 11 (обработка)
- [ ] Выполнена ячейка 12 (скачивание)

### Для обучения LSTM

- [ ] Подготовлены размеченные видео в структуре `normal/` и `defecation/`
- [ ] Выполнены все шаги для инференса (ячейки 1-10)
- [ ] Выполнена ячейка 16 (классы для данных)
- [ ] Выполнена ячейка 17 (класс тренера)
- [ ] В ячейке 18 указан DATA_DIR
- [ ] Выполнена ячейка 18 (подготовка данных) - **долго!**
- [ ] Выполнена ячейка 19 (обучение) - **очень долго!**
- [ ] Выполнена ячейка 20 (оценка)
- [ ] Выполнена ячейка 22 (экспорт)

---

## 📞 Быстрая помощь

| Проблема | Быстрое решение |
|----------|-----------------|
| Не работает GPU | Runtime → Change runtime type → T4 GPU |
| Пропало соединение | Runtime → Run all (Ctrl+F9) |
| Ошибка импорта | Выполнить ячейку 1 заново |
| Зависло выполнение | Runtime → Interrupt execution |
| Нужно больше памяти | Runtime → View resources → Manage sessions |

---

**Версия инструкции:** 2.0  
**Дата обновления:** 2024-01-15
```

Эта инструкция полностью описывает порядок работы с Google Colab блокнотом, все классы и их методы, а также содержит решения типичных проблем. Сохраните её как `INSTRUCTION.md` в корне проекта.
