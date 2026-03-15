# <u> <center> Parameter Efficient Fine Tuning - LoRA & QLoRA </u> </center>

## Agenda

-----

### Fine-Tuning Revisited
- Full Fine-Tuning vs Efficient Adaptation
- Compute, Memory & Cost Challenges
- Introduction to PEFT

### Mathematical Foundations of PEFT
- Matrix Rank & Low-Rank Intuition
- SVD Perspective & Intrinsic Dimension

### LoRA Deep Dive
- Core Architecture (W + A×B)
- Freezing Base Models
- Hyperparameters & Parameter Savings

### Hands-On Demonstration (LoRA)
- Baseline Model Behavior
- Adding LoRA Adapters
- Training Low-Rank Matrices
- Before vs After Evaluation

### QLoRA Explained
- 4-Bit Quantization & NF4
- Double Quantization
- LoRA + Quantization Synergy

### Practical & Deployment Considerations
- Model Selection Strategy
- Adapter Storage & Multi-Domain Use
- Limitations & Best Practices

-----

## Notes:

- Each Token is represented internally by a vector with 12,288 dimensions. 
- Core Idea of Attention
  - look at other tokens 
  - Measure relevance
  - combine info from important tokens
- Each token generates 3 vectors
  - Query (Q)
  - Key (K)
  - Value (V)
- Attention Score Calculation
  - Attention(Q,K,V) 
- L = 12*d^2
- Attention Layer = 4*d^2
- W' = W + (a/r)BA (Low rank matrix)
  - W &rarr; original weight matrix
  - A & B &rarr; Low matrices
  - r &rarr; Rank

  

