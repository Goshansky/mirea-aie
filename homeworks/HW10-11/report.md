# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- **Часть A:** датасет `STL10` (10 классов, 96×96) — стандартный учебный выбор из задания, умеренный размер и явный train/test split.
- **Часть B:** **`Pascal VOC`** (`torchvision.datasets.VOCSegmentation`, `year="2012"`), трек **segmentation** — pretrained `DeepLabV3_ResNet50` с `DeepLabV3_ResNet50_Weights.DEFAULT`; препроцессинг входа: **`DEEPLAB_WEIGHTS.transforms()`** (как рекомендует torchvision для выбранных весов). В колонке `dataset` файла `runs.csv` для V1/V2 указано то же имя: **`Pascal VOC`**.
- **Сравнение в A:** четыре конфигурации C1–C4 (простая CNN без/с аугментациями; ResNet18 head-only и частичный fine-tune `layer4+fc`).
- **Сравнение в B:** два режима постобработки маски — **V1** (argmax → бинарный foreground) и **V2** (удаление мелких связных компонент / морфологическая очистка).

## 2. Среда и воспроизводимость

- Python / torch / torchvision: см. вывод первой ячейки ноутбука или `python --version`, `torch.__version__`, `torchvision.__version__`.
- Устройство в последнем прогоне: зафиксировано в [`./artifacts/best_classifier_config.json`](./artifacts/best_classifier_config.json) (`device`; у автора последнего прогона — `cpu`).
- Seed: **42** (фиксированы `random`, `numpy`, `torch`).
- Как запустить: открыть `homeworks/HW10-11/HW10-11.ipynb`, рабочая директория — `HW10-11`, выполнить **Run All** (данные в `./data/`, артефакты в `./artifacts/`).

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: **STL10**
- Разделение: официальный `train` (5000) и `test` (8000); из `train` отделён **val** как 20% с фиксированным генератором (`VAL_FRAC=0.2`, `seed=42`).
- Базовые transforms: `ToTensor` + нормализация `(0.5, 0.5, 0.5)`.
- Augmentation transforms: `RandomHorizontalFlip`, `RandomCrop(96, padding=8)` + та же нормализация.
- Комментарий: 10 классов, изображения 96×96; задача проще, чем на мелком CIFAR, но без ImageNet-масштаба простая CNN остаётся слабее ResNet с предобучением.

### 3.2. Часть B: structured vision

- Датасет (как в ДЗ и в `runs.csv`): **`Pascal VOC`**, загрузка через `VOCSegmentation`, `image_set="val"` для метрик, подвыборка `VOC_METRIC_N` кадров для ускорения.
- Трек: **segmentation**
- **Класс фона VOC:** `VOC_BACKGROUND_CLASS_ID = 0`. Пиксели **`255`** (ignore/граница) приводим к **`0`**.
- **Foreground для метрик:** бинарная маска **`label > 0`**, т.е. объединение всех объектных классов VOC (всё, что не фон).
- Предсказания: логиты DeepLabV3 `[21, H, W]`; **V1:** argmax → foreground, если **`predicted_class > 0`**.
- Препроцессинг изображений для модели: **`DeepLabV3_ResNet50_Weights.DEFAULT.transforms()`**.

## 4. Часть A: модели и обучение (C1-C4)

- **C1 (simple-cnn-base):** `SimpleCNN`, train без аугментаций.
- **C2 (simple-cnn-aug):** та же сеть, train с аугментациями.
- **C3 (resnet18-head-only):** `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` (в ноутбуке также переменная `RESNET18_WEIGHTS`), заморожен backbone, обучается только `fc`.
- **C4 (resnet18-finetune):** разморожены `layer4` и `fc`, малый LR относительно головы.

Дополнительно:

- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Batch size: **16**
- Epochs: **15** (CNN), **12** (ResNet)
- **Preprocessing ResNet18 (pretrained):** для val/test — **`RESNET18_WEIGHTS.transforms()`**; для train — `RandomResizedCrop(224)` + `RandomHorizontalFlip` + `ToTensor` + `Normalize` с mean/std ImageNet (в коде: из `meta`, если есть ключи `mean`/`std`, иначе классические константы — в части версий torchvision в `meta` их нет).
- Критерий выбора лучшей модели: максимальная **val accuracy** по эпохам (сохраняется лучший checkpoint).
- Ключевые гиперпараметры лучшей модели также продублированы в [`./artifacts/best_classifier_config.json`](./artifacts/best_classifier_config.json) (`learning_rate`, `epochs_trained`, `optimizer`, `loss`, `batch_size`, …).

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран segmentation track

- Модель: **DeepLabV3_ResNet50** (`weights=DeepLabV3_ResNet50_Weights.DEFAULT`).
- **Foreground:** фон — класс **`0`**; после `255→0` считаем **foreground = `(mask > 0)`** (все неконфликтные объектные классы VOC в одной бинарной маске).
- **V1:** softmax по 21 логиту → **argmax** → **foreground, если предсказанный класс > 0** (см. `pred_to_binary` в ноутбуке).
- **V2:** маска **V1** → **`scipy.ndimage.binary_opening`** (структура **3×3**) → удаление связных компонент с площадью **< 400** px; без SciPy — упрощённая морфология **3×3** через max-pool/min-pool в `postprocess_v2`.
- Mean IoU: усреднение **IoU** по бинарным маскам на выбранном подмножестве изображений.
- Дополнительно: **pixel_precision** и **pixel_recall** для бинарного foreground.

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: [`./artifacts/runs.csv`](./artifacts/runs.csv)
- Лучшая модель части A: [`./artifacts/best_classifier.pt`](./artifacts/best_classifier.pt)
- Конфиг лучшей модели части A: [`./artifacts/best_classifier_config.json`](./artifacts/best_classifier_config.json)
- Кривые лучшего прогона классификации: [`./artifacts/figures/classification_curves_best.png`](./artifacts/figures/classification_curves_best.png)
- Сравнение C1-C4: [`./artifacts/figures/classification_compare.png`](./artifacts/figures/classification_compare.png)
- Визуализация аугментаций: [`./artifacts/figures/augmentations_preview.png`](./artifacts/figures/augmentations_preview.png)
- Визуализации второй части: [`./artifacts/figures/segmentation_examples.png`](./artifacts/figures/segmentation_examples.png), [`./artifacts/figures/segmentation_metrics.png`](./artifacts/figures/segmentation_metrics.png)

Короткая сводка по фактическим значениям из [`./artifacts/runs.csv`](./artifacts/runs.csv) (последний прогон):

- **Лучший эксперимент части A:** `C4 (resnet18-finetune)`.
- **C1:** `best_val_accuracy = 0.562`.
- **C2:** `best_val_accuracy = 0.625`.
- **C3:** `best_val_accuracy = 0.937`.
- **C4:** `best_val_accuracy = 0.938`, `test_accuracy = 0.942125` (финальная проверка лучшей модели; совпадает с [`best_classifier_config.json`](./artifacts/best_classifier_config.json)).
- **Эффект аугментаций:** `C2 - C1 = +0.063` по `best_val_accuracy`.
- **Эффект transfer learning:** `C3`/`C4` сильно выше `C1/C2` (рост порядка `+0.31…+0.38` относительно C1 по val).
- **Head-only vs fine-tune:** `C4` лучше `C3` на `+0.001` по `best_val_accuracy`.
- **Segmentation V1:** `mean_iou ≈ 0.7522`, `precision ≈ 0.8076`, `recall ≈ 0.8929`.
- **Segmentation V2:** `mean_iou ≈ 0.7519`, `precision ≈ 0.8087`, `recall ≈ 0.8909`.
- **V1 vs V2:** различия небольшие: IoU почти на уровне, precision чуть вырос у V2, recall чуть ниже; жёсткая постобработка в среднем не «ломает» качество на этом поднаборе.

## 7. Анализ

На STL10 базовая CNN с нуля (C1) заметно уступает transfer learning: это ожидаемо из-за малого объёма train (5000 изображений) и отсутствия сильного визуального prior. Добавление аугментаций в C2 улучшило `best_val_accuracy` с `0.562` до `0.625`, что подтверждает чувствительность к вариативности данных. Наиболее сильный скачок дал переход к pretrained ResNet18: `0.937` (C3) и `0.938` (C4), то есть backbone ImageNet перенёс полезные признаки. Разница между C3 и C4 снова мала (`+0.001` по val), поэтому fine-tune `layer4+fc` даёт лишь небольшой выигрыш относительно head-only.

Во второй части foreground — «все классы VOC кроме фона» (`label > 0` после `255→0`); для этого корректны `mean_iou` и pixel-level `precision`/`recall`. По текущему прогону V1 и V2 близки: IoU почти не меняется, у V2 чуть выше precision и чуть ниже recall — типичный компромисс после морфологии и отсечения мелких компонент без явного «провала» качества на выбранном поднаборе `VOC_METRIC_N`.

## 8. Итоговый вывод

Для STL10 в этом эксперименте лучший конфиг — **C4 (`ResNet18`, `layer4+fc` fine-tune)** с `best_val_accuracy=0.938` и `test_accuracy=0.942125`; **C3** отстаёт на `0.001` по val и остаётся сильным базовым вариантом. Главный вывод по части A: transfer learning даёт основной прирост, аугментации заметно помогают CNN с нуля. Главный вывод по части B: IoU и pixel-level метрики отражают перекрытие маски и шум на фоне; постобработка V2 в этом прогоне лишь слегка перераспределяет precision/recall относительно V1.

## 9. Приложение (опционально)

— (не использовалось.)
