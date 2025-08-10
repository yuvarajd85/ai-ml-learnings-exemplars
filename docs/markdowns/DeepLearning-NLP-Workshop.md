# Deep Learning - NLP Workshop

----

## Deep Learning &rarr; Day-1 &rarr; 08-09-2025

----

## 1. Why Deep Learning ?

Deep Learning (DL) is a subfield of Machine Learning (ML) that uses multi-layered artificial neural networks to model complex patterns in data.

### Key Reasons for Popularity
* Handles raw, unstructured data: Images, audio, text — without heavy manual feature engineering.
* Automatic feature extraction: Learns hierarchical features directly from data.
* Scales with data and compute: Performance improves significantly with more data and GPU/TPU power.
* Breakthrough results: Outperformed traditional ML in computer vision, NLP, speech recognition, recommendation systems, etc.
* End-to-end learning: Goes from raw input → prediction without intermediate handcrafted features.

### Real-world examples:

* Face recognition in smartphones
* Google Translate’s neural machine translation
* ChatGPT and generative AI models
* Autonomous vehicle perception systems

----

## 2. DL vs ML: Technical Differences and Use Cases**

| **Aspect**              | **Machine Learning (ML)**                                | **Deep Learning (DL)**                                           |
| ----------------------- | -------------------------------------------------------- | ---------------------------------------------------------------- |
| **Data requirement**    | Works well with small/medium datasets                    | Requires large datasets for good performance                     |
| **Feature engineering** | Manual feature extraction is crucial                     | Learns features automatically                                    |
| **Model complexity**    | Simpler models (e.g., linear regression, decision trees) | Multi-layer neural networks with millions/billions of parameters |
| **Computation**         | Can run on CPUs easily                                   | Often requires GPUs/TPUs                                         |
| **Interpretability**    | Easier to interpret                                      | More of a “black box”                                            |
| **Performance**         | Can saturate on complex tasks                            | Can scale performance with data and depth                        |

**Use Cases**

* **ML**: Credit scoring, churn prediction, time series forecasting (small data), recommendation with tabular data
* **DL**: Image classification, NLP (chatbots, translation), audio transcription, large-scale recommender systems

---

## **3. Steps Involved in a Deep Learning Workflow**

1. **Define the Problem**

   * Is it classification, regression, segmentation, generation, etc.?

2. **Gather & Prepare Data**

   * Collect datasets (images, text, audio, etc.).
   * Split into training, validation, test sets.
   * Preprocess (normalization, tokenization, data augmentation).

3. **Choose a Model Architecture**

   * CNNs for images, RNN/LSTM/Transformers(Encoder-Decoder Architecture) for sequences(Time Series data, Text Sequences), GANs for generative tasks. Auto Encoders &rarr; Variational Auto Encoders &rarr;  GAN (Generative Adversarial Network).
   * FCNN / DNN &rarr; Fully Connected Neural networks.
   * GRU (Gated Recurrent Units) 
   
    **Rearranged / Reclassified based on Architecture**
    * DNN / FCNN
    * RNN / LSTM
    * Transformers 
    * GANs
    * Auto Encoders / Variational Auto Encoders
    * GRUs

4. **Define the Loss Function**

   * E.g., Cross-Entropy Loss for classification, MSE for regression. KLD &rarr; KL Divergence
   * [Keras Loss Functions](https://keras.io/api/losses/)

5. **Select the Optimizer**

   * SGD, Adam, RMSprop — update weights to minimize loss.

6. **Train the Model**

   * Forward pass → compute loss → backward pass (backpropagation) → update weights.

7. **Validate & Tune Hyperparameters**

   * Learning rate, batch size, number of layers, dropout rate.

8. **Test the Model**

   * Measure generalization on unseen data.

9. **Deploy**

   * Package the model into an application, API, or edge device.

---

## **4. Popular Frameworks for Deep Learning**

| Framework      | Language      | Key Features                                                 | Popular Use                          |
| -------------- | ------------- | ------------------------------------------------------------ | ------------------------------------ |
| **TensorFlow** | Python, C++   | Large ecosystem, production-ready, integrates with Keras     | Google-scale deployments             |
| **PyTorch**    | Python        | Dynamic computation graphs, easy to debug, research-friendly | Academic research, production (Meta) |
| **Keras**      | Python        | High-level API (can run on TensorFlow, Theano, CNTK)         | Fast prototyping                     |
| **JAX**        | Python        | Autograd + XLA compilation for speed                         | High-performance research            |
| **MXNet**      | Python, Scala | Efficient distributed training                               | AWS SageMaker backend                |

📌 **Current trend**: PyTorch dominates research; TensorFlow/Keras still strong in production.

---

## **5. What are Neurons in Deep Learning?**

A **neuron** is the basic computational unit in a neural network — inspired by biological neurons.

### **Structure**

* **Inputs** ($x_1, x_2, ..., x_n$)
* **Weights** ($w_1, w_2, ..., w_n$) → determines importance of each input
* **Bias** ($b$) → shifts activation threshold
* **Summation** → $z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$
* **Activation function** → non-linear transformation (ReLU, sigmoid, tanh, etc.)

📌 **Mathematical representation**:

$$
y = \phi \left( \sum_{i=1}^{n} w_i x_i + b \right)
$$

where $\phi$ is the activation function.

---

## **6. What Do Neurons Learn and How Do They Learn?**

### **What they learn**

* In early layers → **low-level patterns** (edges, curves in images; word associations in text).
* In deeper layers → **high-level concepts** (faces, objects, sentence meaning).

### **How they learn**

* **Forward pass**: Input flows through the network → prediction is made.
* **Loss computation**: Compare prediction to actual label using loss function.
* **Backward pass (Backpropagation)**:

  * Compute gradients of loss w\.r.t. weights (∂Loss/∂w).
  * Update weights using **gradient descent**:

    $$
    w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w}
    $$

    where $\eta$ is the learning rate.
* Repeat for many **epochs** until convergence.

---

### **Mini Python Example**: A Tiny Neural Network in PyTorch

---- 

## Important Pointers 

----
- Loss functions (function of this error which the model is making during the training phase)
 is the actual loss/cost function used to minimize the error made by the model during training process

- Optimizers (in ML solvers) >>> Actual algo which is used to find the minima in the loss function

Most ML Algo use hard-coded Loss functions and optimizers/solver

----

- What is Deep Leaning: all learning happen using neural networks
- Why/When is DL superior to ML?

- Diff neural networks >> are diff configurations/architectures of neurons, specialized for a given task.

- Activation functions

----
- What is Deep Learning, when to use DL vs ML?
- What is Forward Pass? 
- What is Backpropagation? What is it used for ?
- How to decide the number of nodes in the input layer?
- How to decide the number of neurons in the output layer?
- What are activation functions? and where/why are they used?
- How to decide which activation function to be used for what kind of layers 
 	- for hidden layers (best: relu)
	- for o/p layers >>> regression > relu, otherwise linear
	- for o/p layers >>> binary classification >> sigmoid (1 neuron o/p layer) or softmax (2 neurons o/p layer)
	- for o/p layers >>> multiclass classification >> softmax

- what is epoch in model training?
- what batch size in training? how does it affect the training process?
- How to count the total number of parameters defining a NN?
- What are Parameters vs Hyperparameters of a Model (ML/DL)?
- What are Hidden layers? 
- Why do we need HL?
- How do you decide how many HL to be used in a NN?
- How do you decide how many neurons to be used in a HL?
- What is Gradient Descent Algo? Where does it fit in NN discussion?
- What are diff variants of GD? Why do we need diff variants?

-----

- [Python-Machine-Learning-Book](https://github.com/rasbt/python-machine-learning-book-3rd-edition)
- [HandsOn-Machine-Learning](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291)
- [Machine-Learning Book](https://github.com/rasbt/machine-learning-book)
- [Deep-Learning](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438)

---- 

### Activation Functions and Optmizers Links

----

https://www.analyticsvidhya.com/blog/2017/10/fundamentals-deep-learning-activation-functions-when-to-use-them/

https://www.learnopencv.com/understanding-activation-functions-in-deep-learning/

https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

http://www.machineintellegence.com/different-types-of-activation-functions-in-keras/


----


### Optimizers:


https://blog.algorithmia.com/introduction-to-optimizers/

https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f

https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1

https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/

----

### Back Propagation:

https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

http://blog.manfredas.com/backpropagation-tutorial/

https://www.guru99.com/backpropogation-neural-network.html

https://www.edureka.co/blog/backpropagation/

http://neuralnetworksanddeeplearning.com/chap2.html



