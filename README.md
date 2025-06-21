# 🔥 FireClassification

This repository contains 🔥 fire detection and classification tools using machine learning and computer vision.

---

## 📋 Table of Contents

- [Overview](#-overview)  
- [Features](#-features)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Model & Dataset](#-model--dataset)  
- [Evaluation](#-evaluation)  
- [Contributing](#-contributing)  
- [License](#-license)  
- [Contact](#-contact)  

---

## 🔥 Overview

This project aims to automatically detect and classify fire from images (or video frames) using deep learning. It’s designed to help early fire detection for environmental monitoring, safety systems, or industrial use.

---

## ✨ Features

- ⚙️ Classifies whether an image contains fire (yes/no)  
- 📊 Outputs confidence scores  
- 🔄 Easy to extend with new architectures or datasets  
- 🧪 Includes evaluation tools with metrics and visualizations  

---

## 🛠️ Installation

1. Clone the repo:  
   ```bash
   git clone https://github.com/DavidTovmasyan/FireClassification.git
   cd FireClassification
````

2. (Optional) Create a virtual environment:

   ```bash
   python3 -m venv venv && source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Training

```bash
python train.py \
  --data_dir path/to/dataset \
  --epochs 30 \
  --batch_size 16 \
  --model resnet18
```

### 2. Inference

```bash
python predict.py \
  --model_model path/to/checkpoint.pth \
  --image path/to/test.jpg
```

This will print whether the image has fire, along with a confidence score.

---

## 🧠 Model & Dataset

* **Models included:** ResNet, MobileNet, (custom CNN)
* **Dataset:** Public fire/non-fire image datasets (e.g., FiSmo)
  *Override using `--data_dir`*

Include preprocessing details (resize, normalization) and training hyperparameters in `train.py`.

---

## 📈 Evaluation

Evaluate classification performance:

```bash
python evaluate.py \
  --model path/to/checkpoint.pth \
  --data_dir path/to/val_dataset
```

Produces metrics like accuracy, precision, recall, F1, and visual confusion matrix.

---

## 🤝 Contributing

Contributions are welcome! Please follow:

1. Fork
2. Create a feature branch (`git checkout -b feature-x`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push (`git push origin feature-x`)
5. Open a pull request

Include tests/validation and update documentation earlier.

---

## ⚖️ License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📩 Contact

Developed by **David Tovmasyan**.
For questions or collaboration: [email@example.com](mailto:email@example.com)
Feel free to connect via GitHub or LinkedIn.

---

### 📌 Optional Enhancements

* Add badges for build status, PyPI, coverage, GitHub actions
* Provide example images in a gallery (e.g. `examples/` folder)
* Supply a Colab/demo notebook
* Include Docker support
