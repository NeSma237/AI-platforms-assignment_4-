# 👗 Fashion-MNIST Classification Using Keras & PyTorch  
This project provides a fair comparison between the same Convolutional Neural Network (CNN) model implemented twice:  
- once using **TensorFlow/Keras**,  
- and once using **PyTorch**.

The goal is to understand the differences between the two deep learning frameworks in terms of:  
⚡ Ease of use  
⚡ Data handling  
⚡ Training loop structure  
⚡ Final performance on the same dataset

---

## 📌 Dataset
The dataset used is **Fashion-MNIST**, available directly via Keras.

It contains:
- 60,000 training images  
- 10,000 testing images  
- Image size: 28×28×1 (grayscale)  
- 10 clothing categories (T-shirt, Bag, Sneakers, etc.)

---

## 🧠 Model Architecture (Identical in Both Keras & PyTorch)

A simplified **LeNet-5** architecture was used:

- Conv2D: 6 filters (5×5)  
- MaxPooling (2×2)  
- Conv2D: 16 filters (5×5)  
- MaxPooling (2×2)  
- Flatten  
- Dense: 120  
- Dense: 84  
- Dense: 10 (softmax)

> This ensures a **fair comparison** between the two frameworks.

---

## 🚀 1. Keras Implementation
### ✨ Features:
- Uses `model.fit()` directly  
- No DataLoader needed  
- Supports built-in callbacks like:
  - EarlyStopping  
  - ModelCheckpoint  
  - ReduceLROnPlateau  

### ▶️ Run the Keras code:
```bash
pip install tensorflow matplotlib
python keras_model.py
```

### ✔ Key part of the architecture:
```python
model = Sequential([
    Input(shape=(28,28,1)),
    layers.Conv2D(6, (5,5), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(16, (5,5), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

---

## 🐍 2. PyTorch Implementation
### ✨ Why PyTorch requires more steps?
PyTorch gives **more manual control**, so you must:
- Convert NumPy arrays to tensors  
- Add channel dimension  
- Create Dataset and DataLoader  
- Write the training loop manually  
- Handle device placement (CPU/GPU)

### ▶️ Run the PyTorch code:
```bash
pip install torch matplotlib
python pytorch_model.py
```

### ✔ Key part of the training loop:
```python
for epoch in range(num_epochs):
    net.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## 📊 Keras vs PyTorch (High-Level Comparison)

| Feature | Keras | PyTorch |
|---------|-------|---------|
| Data handling | Automatic | Manual |
| Training process | `model.fit()` | Custom training loop |
| Callbacks | Built-in (easy) | Manual implementation |
| Flexibility | Less | Very high |
| Difficulty | Easier | More coding |

---

## 📉 Results
Both implementations:
- Trained for **13 epochs**
- Used the **same architecture**
- Used the **same batch size (128)**  
- Plotted both training and test loss curves.

### Plotting loss in Keras:
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
```

### Plotting loss in PyTorch:
```python
plt.plot(pytorch_train_losses)
plt.plot(pytorch_test_losses)
```

---

## 🛠️ Requirements

```
torch
torchvision
tensorflow
numpy
matplotlib
opencv-python
```

Install all dependencies:

```bash
pip install torch torchvision tensorflow numpy matplotlib opencv-python
```

---

## 📂 Recommended Project Structure

```
project/
│
├── keras vs. pytorch.ipynb.py
├── README.md
└── best_model.weights.h5
```

---

## 🏁 Conclusion
This project provides a clear comparison between:
- The **simplicity of Keras** for quick model development  
- The **flexibility and control of PyTorch** for research-level experiments  

Both models use the same architecture, same data, and same settings to ensure fairness.

---

## ✨ Author
**Nesma Nasser Galal Hasan**

---
