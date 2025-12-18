# 🔢 Handwritten Digit Classifier

A deep learning web application that recognizes handwritten digits (0-9) using a Convolutional Neural Network trained on the MNIST dataset.

![Demo](https://img.shields.io/badge/Accuracy-98%25-brightgreen)

## 🎯 Features

- Real-time digit recognition with 98% accuracy
- Interactive web interface for drawing digits
- Built with PyTorch and Gradio
- Trained on 60,000 MNIST images

## 🚀 Live Demo

[Try it here](#) _(Coming soon - will deploy to Hugging Face Spaces)_

## 📸 Screenshot

![Accuracy](screenshotAccuracy.png)
![Demo](screenshot2.png)

## 🛠️ Technology Stack

- **Framework**: PyTorch
- **Model**: Convolutional Neural Network (CNN)
- **Frontend**: Gradio
- **Dataset**: MNIST (28x28 grayscale images)

## 🏗️ Model Architecture
```
- Conv2D Layer (1 → 32 channels, 3x3 kernel)
- Conv2D Layer (32 → 64 channels, 3x3 kernel)
- MaxPooling (2x2)
- Dropout (25%)
- Fully Connected (9216 → 128)
- Dropout (50%)
- Output Layer (128 → 10 classes)
```

## 📊 Performance

- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98%
- **Training Time**: ~3 minutes on CPU
- **Model Size**: ~1.5 MB

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/abhinawagoo/digit-classifier.git
cd digit-classifier
```

2. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Train the model (optional - pre-trained model included):
Open `digit_classifier.ipynb` in Jupyter and run all cells.

### Run the web app:
```bash
python app.py
```

Then open http://127.0.0.1:7860 in your browser.

## 📁 Project Structure
```
digit-classifier/
├── digit_classifier.ipynb   # Training notebook
├── app.py                    # Gradio web app
├── digit_classifier.pth      # Trained model weights
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── data/                     # MNIST dataset (auto-downloaded)
```

## 🧠 What I Learned

- Implementing CNNs from scratch in PyTorch
- Training deep learning models with backpropagation
- Preventing overfitting using dropout
- Building interactive ML demos with Gradio
- Deploying ML models as web applications

## 🔮 Future Improvements

- [ ] Deploy to Hugging Face Spaces
- [ ] Add support for real-time webcam digit recognition
- [ ] Implement data augmentation for better generalization
- [ ] Add confidence visualization
- [ ] Support for custom digit datasets

## 📝 License

MIT License - feel free to use this project for learning!

## 🤝 Connect

Feel free to reach out if you have questions or suggestions!

- GitHub: [@YOUR_USERNAME](https://github.com/abhinawagoo)
```

## Step 5: Create .gitignore

Create a file called `.gitignore` (to avoid uploading unnecessary files):
```
# Virtual environment
.venv/
venv/
env/

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data
data/

# macOS
.DS_Store

# IDE
.vscode/
.idea/

# Model checkpoints (optional - include if you want to share trained model)
# *.pth