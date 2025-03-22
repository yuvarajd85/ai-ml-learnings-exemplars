### **PyTorch Tutorial for Beginners: From Scratch**  
PyTorch is an open-source deep learning framework widely used for **AI research, NLP, computer vision, and LLM engineering**. This tutorial will help you get started with **PyTorch from scratch**, covering **tensors, autograd, neural networks, and training a simple model**.

---

## **1️⃣ Installing PyTorch**
Before using PyTorch, install it with the following command:  
```bash
pip install torch torchvision torchaudio
```
You can also check if PyTorch is installed correctly by running:
```python
import torch
print(torch.__version__)
```

---

## **2️⃣ Understanding Tensors in PyTorch**
Tensors are the fundamental data structures in PyTorch, similar to **NumPy arrays** but with GPU support.

### **Creating Tensors**
```python
import torch

# Creating a 1D Tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(x)

# Creating a Random Tensor
random_tensor = torch.rand(2, 3)
print(random_tensor)

# Creating a Zeros Tensor
zeros_tensor = torch.zeros(3, 3)
print(zeros_tensor)
```

### **Tensor Operations**
```python
a = torch.tensor([2, 3])
b = torch.tensor([4, 5])

# Element-wise addition
print(a + b)

# Matrix multiplication
mat1 = torch.rand(2, 3)
mat2 = torch.rand(3, 2)
result = torch.mm(mat1, mat2)
print(result)

# Reshaping tensors
x = torch.rand(4, 4)
x_reshaped = x.view(2, 8)  # Reshape to (2x8)
print(x_reshaped)
```

---

## **3️⃣ PyTorch Autograd (Automatic Differentiation)**
PyTorch has **autograd** to automatically compute gradients for deep learning.

```python
x = torch.tensor(2.0, requires_grad=True)  # Enable gradient tracking
y = x ** 3  # Compute function

y.backward()  # Compute gradient dy/dx
print(x.grad)  # Should print 3*x^2 = 12
```

---

## **4️⃣ Creating a Simple Neural Network in PyTorch**
PyTorch provides the `torch.nn` module for building neural networks.

### **Step 1: Define the Model**
```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer (2 neurons) → Hidden layer (4 neurons)
        self.fc2 = nn.Linear(4, 1)  # Hidden layer (4 neurons) → Output layer (1 neuron)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)  # Output layer
        return x

model = SimpleNN()
print(model)
```

---

## **5️⃣ Training a Model with PyTorch**
We'll train a simple neural network using **gradient descent**.

### **Step 1: Prepare Data**
```python
import torch.optim as optim

# Sample dataset (inputs & labels)
X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=torch.float32)
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### **Step 2: Train the Model**
```python
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear gradients
    predictions = model(X)  # Forward pass
    loss = loss_function(predictions, Y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')
```

---

## **6️⃣ Saving & Loading Models**
To save and load a trained model:
```python
# Save model
torch.save(model.state_dict(), "simple_nn.pth")

# Load model
model.load_state_dict(torch.load("simple_nn.pth"))
```

---

## **7️⃣ Running the Model on GPU**
To enable GPU training:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X = X.to(device)
Y = Y.to(device)
```

---

### **✅ Summary**
| Concept | Code Example |
|---------|-------------|
| Create Tensor | `torch.tensor([1,2,3])` |
| Matrix Multiplication | `torch.mm(A, B)` |
| Compute Gradients | `y.backward()` |
| Define Model | `class SimpleNN(nn.Module)` |
| Train Model | `loss.backward(), optimizer.step()` |
| Save Model | `torch.save(model.state_dict(), "file.pth")` |
| Load Model | `model.load_state_dict(torch.load("file.pth"))` |
| Use GPU | `model.to("cuda")` |

---

### **🚀 Next Steps**
- Implement a **CNN for image classification** (MNIST).
- Train a **transformer model** for NLP using Hugging Face.
- Explore **TorchServe for model deployment**.

Would you like a **hands-on project**, like training a PyTorch NLP model? 😊