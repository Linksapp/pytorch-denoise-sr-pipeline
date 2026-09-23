# PyTorch Denoise + Super-Resolution Pipeline

PyTorch-пайплайн для восстановления изображений с использованием **noise estimation**, **image denoising** и **super-resolution**.

Проект исследует каскадный подход, при котором изображение сначала анализируется и очищается от шума, а затем восстанавливается до более высокого разрешения.

## Architecture

```text
Input Image
     │
     ▼
Noise Estimation
     │
     ▼
Image Denoising
     │
     ▼
Super-Resolution
     │
     ▼
Restored Image
```

Основные компоненты:

* **Noise Estimator** — оценивает характеристики шума;
* **Denoiser** — удаляет шум с использованием residual CNN и attention mechanisms;
* **Upscaler** — выполняет super-resolution с использованием residual blocks и PixelShuffle;
* **Dataset Pipeline** — подготавливает данные для обучения;
* **Evaluation** — оценивает качество восстановления.

## Blind Mode

Проект поддерживает **blind image restoration** — режим, в котором параметры шума заранее неизвестны.

Вместо передачи известного уровня шума pipeline автоматически оценивает его с помощью `NoiseEstimator`, формирует noise map и передаёт её denoising модели.

```text
Real / Unknown Image
        │
        ▼
  Noise Estimator
        │
        ▼
    Noise Map
        │
        ▼
     Denoiser
        │
        ▼
     Upscaler
        │
        ▼
Restored Image
```

Blind режим позволяет применять pipeline к **реальным изображениям**, где характеристики шума заранее неизвестны и не могут быть заданы вручную.

## Denoising

Denoiser основан на residual CNN architecture и поддерживает:

* residual connections;
* channel attention;
* spatial attention;
* CBAM;
* dilated convolutions;
* residual scaling.

Noise map может использоваться как дополнительный входной канал для адаптации denoising к оценённым характеристикам изображения.

## Super-Resolution

Super-resolution модель восстанавливает пространственное разрешение изображения с использованием residual blocks и `PixelShuffle`.

```text
Feature Extraction
        │
        ▼
Residual Blocks
        │
        ▼
Reconstruction
        │
        ▼
PixelShuffle
        │
        ▼
High-Resolution Image
```

## Dataset & Training

Для обучения используются PyTorch datasets с поддержкой:

* downsampling;
* synthetic noise generation;
* random crops;
* data augmentation;
* denoising;
* super-resolution;
* noise estimation;
* cascade training.

Для blind режима pipeline не требует заранее известного уровня шума на входном изображении.

## Evaluation

Качество восстановления оценивается с помощью:

* **MSE** — пиксельная ошибка;
* **PSNR** — отношение сигнала к шуму;
* **SSIM** — структурное сходство изображений.

## Technology Stack

* Python
* PyTorch
* TorchVision
* NumPy
* OpenCV
* Matplotlib
* Pandas
* Jupyter Notebook

## Project Structure

```text
pytorch-denoise-sr-pipeline/
│
├── datasets.py
├── enchaned_denoiser.py
├── estimator.py
├── upscaler.py
├── metrics.py
├── model_name.py
│
├── super-res.ipynb
├── cascade.ipynb
├── cascade_blind.ipynb
├── noise-generator.ipynb
└── vis.ipynb
```

## Status

**Research prototype**

Проект предназначен для исследования image restoration и объединения **noise estimation**, **denoising** и **super-resolution** в едином PyTorch pipeline, включая **blind restoration для работы с реальными изображениями с неизвестными характеристиками шума**.
