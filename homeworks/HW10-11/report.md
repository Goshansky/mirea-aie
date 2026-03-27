# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- **Часть A:** датасет `STL10` (10 классов, 96×96) — стандартный учебный выбор из задания, умеренный размер и явный train/test split.
- **Часть B:** `Pascal VOC 2012` (сегментация через `VOCSegmentation`), трек **segmentation** — одна pretrained-модель `DeepLabV3_ResNet50` с весами `COCO_WITH_VOC_LABELS_V1` (21 класс, согласован с VOC).
- **Сравнение в A:** четыре конфигурации C1–C4 (простая CNN без/с аугментациями; ResNet18 head-only и частичный fine-tune `layer4+fc`).
- **Сравнение в B:** два режима постобработки маски — **V1** (argmax → бинарный foreground) и **V2** (удаление мелких связных компонент / морфологическая очистка).

## 2. Среда и воспроизводимость

- Python: см. вывод первой ячейки ноутбука или `python --version`.
- torch / torchvision: см. `import torch; print(torch.__version__)`, `torchvision.__version__`.
- Устройство (CPU/GPU): печатается в ноутбуке как `device:`.
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

- Датасет: **Pascal VOC 2012** (`VOCSegmentation`, split `val` для метрик, подвыборка `VOC_METRIC_N` кадров для ускорения).
- Трек: **segmentation**
- Ground truth: маска VOC; пиксели **255** (границы) приведены к фону **0**; бинарный **foreground** = все классы **> 0**.
- Предсказания: логиты DeepLabV3 `[21, H, W]`, класс по argmax; бинарная маска: foreground, если класс > 0.
- Комментарий: VOC — стандартная разметка сегментации; веса с VOC-лейблами дают осмысленное сопоставление с GT без ручного маппинга COCO→VOC.

## 4. Часть A: модели и обучение (C1-C4)

- **C1 (simple-cnn-base):** `SimpleCNN`, train без аугментаций.
- **C2 (simple-cnn-aug):** та же сеть, train с аугментациями.
- **C3 (resnet18-head-only):** `ResNet18` (`IMAGENET1K_V1`), заморожен backbone, обучается только `fc`.
- **C4 (resnet18-finetune):** разморожены `layer4` и `fc`, малый LR относительно головы.

Дополнительно:

- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Batch size: **16**
- Epochs: **15** (CNN), **12** (ResNet)
- Критерий выбора лучшей модели: максимальная **val accuracy** по эпохам (сохраняется лучший checkpoint).

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран segmentation track

- Модель: **DeepLabV3_ResNet50** (`DeepLabV3_ResNet50_Weights.DEFAULT`).
- Что считается foreground: все пиксели GT/предсказания с классом **не 0** (после обработки границ 255→0).
- **V1:** базовая постобработка — argmax по 21 классу, затем бинаризация (foreground = класс > 0).
- **V2:** альтернативная постобработка — удаление слишком мелких связных компонент (порог площади) после `binary_opening` при наличии SciPy; иначе упрощённая морфология через max-pool.
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

Короткая сводка по фактическим значениям из `runs.csv`:

- **Лучший эксперимент части A:** `C4 (resnet18-finetune)`.
- **C1:** `best_val_accuracy = 0.562`.
- **C2:** `best_val_accuracy = 0.625`.
- **C3:** `best_val_accuracy = 0.942`.
- **C4:** `best_val_accuracy = 0.943`, `test_accuracy = 0.9405` (финальная проверка лучшей модели).
- **Эффект аугментаций:** `C2 - C1 = +0.063` по `best_val_accuracy`.
- **Эффект transfer learning:** `C3` и `C4` сильно выше `C1/C2` (рост примерно на `+0.317...+0.381` относительно C1).
- **Head-only vs fine-tune:** `C4` чуть лучше `C3` на `+0.001` по `best_val_accuracy`.
- **Segmentation V1:** `mean_iou = 0.72298`, `precision = 0.80464`, `recall = 0.86054`.
- **Segmentation V2:** `mean_iou = 0.71994`, `precision = 0.79350`, `recall = 0.85554`.
- **V1 vs V2:** в этом запуске V2 дал небольшую просадку по всем трём метрикам, т.е. выбранная постобработка оказалась слишком агрессивной для текущего поднабора.

## 7. Анализ

На STL10 базовая CNN с нуля (C1) заметно уступает transfer learning: это ожидаемо из-за малого объёма train (5000 изображений) и отсутствия сильного визуального prior. Добавление аугментаций в C2 улучшило `best_val_accuracy` с `0.562` до `0.625`, что подтверждает, что модель была чувствительна к вариативности данных. Наиболее сильный скачок дал переход к pretrained ResNet18: `0.942` (C3) и `0.943` (C4), то есть backbone ImageNet перенёс полезные признаки, которые трудно выучить с нуля на таком размере выборки. Разница между C3 и C4 минимальная (`+0.001`), поэтому в этой постановке fine-tune `layer4+fc` почти не изменил качество относительно head-only.

Во второй части foreground определён как «все классы VOC кроме фона», и под такую бинаризацию корректно считать `mean_iou`, а также pixel-level `precision/recall`. В режиме V1 модель дала `mean_iou=0.72298`, `precision=0.80464`, `recall=0.86054`. Режим V2 (очистка мелких компонент) показал небольшое ухудшение (`mean_iou=0.71994`, `precision=0.79350`, `recall=0.85554`), что говорит о потере части истинно-положительных пикселей вместе с шумом. То есть гипотеза «более жёсткая постобработка улучшит качество» для этого запуска не подтвердилась; логично пробовать меньший порог фильтрации компонент или более мягкую морфологию.

## 8. Итоговый вывод

Для STL10 в этом эксперименте лучший конфиг — **C4 (`ResNet18`, `layer4+fc` fine-tune)** с `best_val_accuracy=0.943` и `test_accuracy=0.9405`, но по факту C3 почти эквивалентен и дешевле по обучению. Главный вывод по части A: transfer learning дал ключевой прирост, а аугментации для CNN с нуля тоже важны и стабильны. Главный вывод по части B: для сегментации нужно оценивать IoU и pixel-level метрики, а постобработка не всегда улучшает результат — её параметры надо подбирать по валидации, а не «вслепую».

## 9. Приложение (опционально)

— (не использовалось.)
