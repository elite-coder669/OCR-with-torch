# OCR-with-torch 🔍

A high-performance Convolutional Neural Network (CNN) built with PyTorch, designed to recognize handwritten characters across 62 distinct classes.

## 🚀 About the Project
This model is capable of identifying handwritten text with high precision, supporting:
* **Uppercase Letters:** A-Z
* **Lowercase Letters:** a-z
* **Digits:** 0-9

The architecture is optimized to handle the nuances of varied handwriting styles, achieving a robust **95% accuracy rate** on the test dataset.

## 📊 Key Features
* **62-Class Classification:** Full alphanumeric support.
* **High Accuracy:** 95% success rate in character recognition.
* **PyTorch Backend:** Optimized for GPU acceleration and flexibility.
* **Pre-processing Pipeline:** Includes image normalization and noise reduction for real-world handwritten samples.

---

## 🛠 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/elite-coder669/OCR-with-torch.git](https://github.com/elite-coder669/OCR-with-torch.git)
   cd OCR-with-torch

```

2. **Set up a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

## 🏗 Model Architecture

The project utilizes a custom **Convolutional Neural Network (CNN)**. Key layers include:

* Multiple `Conv2d` layers for feature extraction.
* `MaxPool2d` for spatial dimensionality reduction.
* `Dropout` layers to prevent overfitting.
* `Linear` fully connected layers leading to a 62-way Softmax output.

---

## 📜 License & Attribution

This project is licensed under the **MIT License**.

### How to Attribute

If you use this project in your own work, research, or applications, please provide attribution to the original author:

**Author:** [Mallupeddi Vamsi Krishna](https://github.com/elite-coder669)

**Project Link:** [https://github.com/elite-coder669/OCR-with-torch](https://www.google.com/search?q=https://github.com/elite-coder669/OCR-with-torch)

> *“Innovation is better when shared, but credit makes it sustainable.”*
