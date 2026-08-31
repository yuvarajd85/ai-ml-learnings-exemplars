# ML Mathematics Workbook: Vectors, Matrices & Determinants
### A Step-by-Step Self-Study Guide for Students

> Every operation is shown manually first, then verified with Python.
> Read the math, trace through it by hand, then run the code.

---

## Table of Contents

1. [Vectors — The Building Block](#1-vectors--the-building-block)
2. [Vector Operations](#2-vector-operations)
3. [Dot Product & Angle Between Vectors](#3-dot-product--angle-between-vectors)
4. [Vector Norms (Measuring Length)](#4-vector-norms-measuring-length)
5. [Matrices — Organizing Numbers](#5-matrices--organizing-numbers)
6. [Matrix Operations](#6-matrix-operations)
7. [Matrix Multiplication — Step by Step](#7-matrix-multiplication--step-by-step)
8. [Transpose of a Matrix](#8-transpose-of-a-matrix)
9. [Special Matrices](#9-special-matrices)
10. [Determinants — Does the Matrix Stretch or Flip?](#10-determinants--does-the-matrix-stretch-or-flip)
11. [Determinant of a 3×3 Matrix (Cofactor Expansion)](#11-determinant-of-a-3x3-matrix-cofactor-expansion)
12. [Trace of a Matrix](#12-trace-of-a-matrix)
13. [Matrix Rank — How Much Information?](#13-matrix-rank--how-much-information)
14. [Matrix Inverse — Undoing a Transformation](#14-matrix-inverse--undoing-a-transformation)
15. [Solving a Linear System Ax = b](#15-solving-a-linear-system-ax--b)
16. [Putting It All Together — A Full ML Story](#16-putting-it-all-together--a-full-ml-story)

---

## 1. Vectors — The Building Block

### What Is a Vector?

A **vector** is simply an ordered list of numbers. Each number is called a **component** or **element**.

```
Column vector (written vertically):      Row vector (written horizontally):

    ┌ 3 ┐                                    [ 3   1   -2 ]
v = │ 1 │    ← 3 components, in R³
    └-2 ┘
```

**Think of it this way:**
- 1 number → a point on a number line (1D)
- 2 numbers → a point on a flat plane (2D)
- 3 numbers → a point in 3D space
- n numbers → a point in n-dimensional space (ML lives here!)

### Real-World Meaning in ML

A house described by 3 features:

```
feature vector for House #1:

    ┌ 1500 ┐   ← square footage
h = │    3 │   ← number of bedrooms
    └    2 ┘   ← number of bathrooms
```

Every row in your dataset IS a vector.

### Python

```python
import numpy as np

# Define vectors
v = np.array([3, 1, -2])        # column vector (stored as 1D array)
h = np.array([1500, 3, 2])      # house feature vector

print("Vector v:", v)
print("Shape:", v.shape)        # (3,) — 3 components

# Access individual elements (0-indexed)
print("First component v[0]:", v[0])   # 3
print("Third component v[2]:", v[2])   # -2
```

---

## 2. Vector Operations

### 2.1 Vector Addition

Add **component by component**. Vectors must have the same number of components.

```
    ┌ 2 ┐       ┌ 5 ┐       ┌ 2+5 ┐   ┌  7 ┐
a = │ 4 │   b = │-1 │   a+b=│ 4-1 │ = │  3 │
    └-3 ┘       └ 6 ┘       └-3+6 ┘   └  3 ┘
```

**Step by step:**
- Row 1: 2 + 5 = **7**
- Row 2: 4 + (-1) = **3**
- Row 3: -3 + 6 = **3**

**Geometric meaning:** Place vector b at the tip of vector a. The result is where you end up.

### 2.2 Scalar Multiplication

Multiply **every component** by the scalar (a single number).

```
        ┌ 2 ┐       ┌ 3×2 ┐   ┌ 6 ┐
3 × a = │ 4 │   =   │ 3×4 │ = │12 │
        └-3 ┘       │ 3×-3│   └-9 ┘
```

**Geometric meaning:** Scales the vector length. Negative scalar reverses direction.

### 2.3 Vector Subtraction

```
        ┌ 2 ┐   ┌ 5 ┐   ┌ 2-5 ┐   ┌-3 ┐
a - b = │ 4 │ - │-1 │ = │ 4+1 │ = │ 5 │
        └-3 ┘   └ 6 ┘   └-3-6 ┘   └-9 ┘
```

### Python

```python
import numpy as np

a = np.array([2, 4, -3])
b = np.array([5, -1, 6])

# Addition
print("a + b =", a + b)         # [ 7  3  3]

# Scalar multiplication
print("3 * a =", 3 * a)         # [ 6 12 -9]

# Subtraction
print("a - b =", a - b)         # [-3  5 -9]

# Linear combination: 2a - 3b
combo = 2 * a - 3 * b
print("2a - 3b =", combo)       # [2*2-3*5, 2*4-3*(-1), 2*(-3)-3*6]
                                 # = [4-15, 8+3, -6-18] = [-11, 11, -24]
```

---

## 3. Dot Product & Angle Between Vectors

### What Is the Dot Product?

Multiply matching components, then **sum** all those products.

```
        ┌ 2 ┐   ┌ 5 ┐
a · b = │ 4 │ · │-1 │  =  (2×5) + (4×-1) + (-3×6)
        └-3 ┘   └ 6 ┘

     =  10   +  (-4)  +  (-18)

     =  -12
```

**Step by step:**
- Pair 1: 2 × 5 = 10
- Pair 2: 4 × (-1) = -4
- Pair 3: (-3) × 6 = -18
- Sum: 10 + (-4) + (-18) = **-12**

### The Angle Formula

The dot product connects to the angle θ between two vectors:

```
a · b = ||a|| × ||b|| × cos(θ)

          a · b
cos(θ) = ————————
         ||a|| ||b||
```

Where `||a||` means the **length** (norm) of vector a.

**Why this matters:**
- cos(θ) = 1 → θ = 0° → vectors point in **same direction**
- cos(θ) = 0 → θ = 90° → vectors are **perpendicular (orthogonal)**
- cos(θ) = -1 → θ = 180° → vectors point in **opposite directions**

**Cosine similarity** is used in recommendation systems, NLP, and search engines — it is just this formula!

### Manual Calculation Example

```
a = [1, 0]    b = [0, 1]   (perpendicular arrows on a graph)

a · b = (1×0) + (0×1) = 0

||a|| = √(1² + 0²) = 1
||b|| = √(0² + 1²) = 1

cos(θ) = 0 / (1 × 1) = 0  →  θ = 90°  ✓
```

### Python

```python
import numpy as np

a = np.array([2.0, 4.0, -3.0])
b = np.array([5.0, -1.0, 6.0])

# Dot product (three equivalent ways)
dot1 = np.dot(a, b)
dot2 = a @ b                  # @ operator (Python 3.5+)
dot3 = np.sum(a * b)          # manual: multiply then sum

print("Dot product:", dot1)   # -12.0

# Angle between a and b
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)

cos_theta = dot1 / (norm_a * norm_b)
theta_rad = np.arccos(np.clip(cos_theta, -1, 1))
theta_deg = np.degrees(theta_rad)

print(f"||a|| = {norm_a:.4f}")
print(f"||b|| = {norm_b:.4f}")
print(f"cos(θ) = {cos_theta:.4f}")
print(f"θ = {theta_deg:.2f}°")

# Cosine similarity (used in NLP / recommender systems)
cos_sim = dot1 / (norm_a * norm_b)
print(f"Cosine similarity: {cos_sim:.4f}")
```

---

## 4. Vector Norms (Measuring Length)

A **norm** is a way to measure the "size" or "length" of a vector.

### L2 Norm (Euclidean Length) — Most Common

```
         ___________________
||v||₂ = √ v₁² + v₂² + v₃²

         _______________
       = √ 3² + 1² + (-2)²

         ___________
       = √ 9 + 1 + 4

         ___
       = √14  ≈ 3.742
```

### L1 Norm (Manhattan Distance)

```
||v||₁ = |v₁| + |v₂| + |v₃|
       = |3| + |1| + |-2|
       = 3 + 1 + 2
       = 6
```

Think: how far to walk on a grid (only right/left/up/down moves allowed).

### L∞ Norm (Max Norm)

```
||v||∞ = max(|v₁|, |v₂|, |v₃|)
       = max(3, 1, 2)
       = 3
```

### Why Different Norms?

| Norm | ML Use |
|------|--------|
| L2 | Ridge regression penalty, Euclidean distance |
| L1 | Lasso regression (promotes sparsity — sets weights to 0) |
| L∞ | Robust optimization, adversarial ML |

### Unit Vector (Normalized)

A unit vector has length 1. Normalize by dividing by the L2 norm:

```
         v       ┌ 3 ┐          ┌ 3/√14 ┐   ┌ 0.802 ┐
v̂ = ———————— =  │ 1 │ / √14  = │ 1/√14 │ = │ 0.267 │
       ||v||₂    └-2 ┘          └-2/√14 ┘   └-0.535 ┘
```

### Python

```python
import numpy as np

v = np.array([3.0, 1.0, -2.0])

l1 = np.linalg.norm(v, 1)     # 6.0
l2 = np.linalg.norm(v, 2)     # 3.7417...
linf = np.linalg.norm(v, np.inf)  # 3.0

print(f"L1 norm: {l1}")
print(f"L2 norm: {l2:.4f}")
print(f"L∞ norm: {linf}")

# Normalize to unit vector
v_hat = v / l2
print("Unit vector:", v_hat)
print("Check length = 1:", np.linalg.norm(v_hat))  # 1.0
```

---

## 5. Matrices — Organizing Numbers

### What Is a Matrix?

A **matrix** is a rectangular grid of numbers arranged in **rows** and **columns**.

```
A 3×4 matrix (3 rows, 4 columns):

        col1  col2  col3  col4
row1  ┌   2    -1    0    5  ┐
row2  │   3     4   -2    1  │
row3  └   0     7    6   -3  ┘
```

We write the size as **m × n** (m rows, n columns).

**Element notation:** `A[i][j]` or `Aᵢⱼ` = element at row i, column j (1-indexed in math, 0-indexed in Python).

```
For the matrix above:
A[1][1] = 2    A[1][4] = 5
A[2][2] = 4    A[3][3] = 6
```

### Why Matrices in ML?

```
Dataset with m=4 samples, n=3 features:

         area  beds  baths
house1 ┌ 1500    3     2  ┐
house2 │  900    2     1  │
house3 │ 2200    4     3  │
house4 └ 1100    2     1  ┘

This IS a 4×3 matrix. Every dataset is a matrix!
```

### Python

```python
import numpy as np

# Create a 3×4 matrix
A = np.array([
    [ 2, -1,  0,  5],
    [ 3,  4, -2,  1],
    [ 0,  7,  6, -3]
])

print("Shape:", A.shape)      # (3, 4) → 3 rows, 4 columns
print("Rows:", A.shape[0])    # 3
print("Columns:", A.shape[1]) # 4

# Access elements (0-indexed in Python)
print("A[0,0] =", A[0, 0])   # 2   (row 1, col 1 in math)
print("A[1,1] =", A[1, 1])   # 4   (row 2, col 2 in math)

# Access full row or column
print("Row 1:", A[0, :])      # [2 -1  0  5]
print("Col 2:", A[:, 1])      # [-1  4  7]

# Dataset example
dataset = np.array([
    [1500, 3, 2],
    [ 900, 2, 1],
    [2200, 4, 3],
    [1100, 2, 1]
])
print("\nDataset shape:", dataset.shape)  # (4, 3)
```

---

## 6. Matrix Operations

### 6.1 Matrix Addition

Same shape required. Add **element by element**.

```
    ┌ 1  2 ┐       ┌ 5  6 ┐       ┌ 1+5  2+6 ┐   ┌  6   8 ┐
A = │ 3  4 │   B = │ 7  8 │   A+B=│ 3+7  4+8 │ = │ 10  12 │
    └ 5  6 ┘       └ 9 10 ┘       └ 5+9 6+10 ┘   └ 14  16 ┘
```

Step by step for each position:
- (0,0): 1+5 = **6**
- (0,1): 2+6 = **8**
- (1,0): 3+7 = **10**
- (1,1): 4+8 = **12**
- (2,0): 5+9 = **14**
- (2,1): 6+10 = **16**

### 6.2 Scalar × Matrix

Multiply **every element** by the scalar.

```
        ┌ 1  2 ┐   ┌ 3×1  3×2 ┐   ┌ 3   6 ┐
3 × A = │ 3  4 │ = │ 3×3  3×4 │ = │ 9  12 │
        └ 5  6 ┘   └ 3×5  3×6 ┘   └15  18 ┘
```

### Python

```python
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[5, 6], [7, 8], [9, 10]])

print("A + B =\n", A + B)
print("A - B =\n", A - B)
print("3 * A =\n", 3 * A)
print("A / 2 =\n", A / 2)

# Element-wise multiplication (NOT matrix multiply!)
print("A * B (element-wise) =\n", A * B)
```

---

## 7. Matrix Multiplication — Step by Step

This is the most important operation. **Not** element-wise — it is a structured combination.

### The Rule

To multiply **A (m×n)** by **B (n×p)**:
- Inner dimensions must match: A's columns = B's rows
- Result has shape **m × p**

```
A is 2×3,  B is 3×2  →  AB is 2×2   ✓ (3 matches 3)
A is 2×3,  B is 2×3  →  CANNOT      ✗ (3 ≠ 2)
```

### How Each Element Is Computed

**Each element `(AB)[i,j]` = dot product of row i of A with column j of B.**

```
    ┌ 1  2  3 ┐       ┌ 7   8 ┐
A = │          │   B = │ 9  10 │
    └ 4  5  6 ┘       └11  12 ┘
```

**Result AB is 2×2.**

**Element (0,0): Row 0 of A · Column 0 of B**
```
[1, 2, 3] · [7, 9, 11] = (1×7) + (2×9) + (3×11)
                        =   7  +   18   +   33
                        =  58
```

**Element (0,1): Row 0 of A · Column 1 of B**
```
[1, 2, 3] · [8, 10, 12] = (1×8) + (2×10) + (3×12)
                         =   8  +    20   +   36
                         =  64
```

**Element (1,0): Row 1 of A · Column 0 of B**
```
[4, 5, 6] · [7, 9, 11] = (4×7) + (5×9) + (6×11)
                        =  28  +   45   +   66
                        = 139
```

**Element (1,1): Row 1 of A · Column 1 of B**
```
[4, 5, 6] · [8, 10, 12] = (4×8) + (5×10) + (6×12)
                         =  32  +    50   +   72
                         = 154
```

**Final result:**
```
        ┌  58   64 ┐
AB  =   │           │
        └ 139  154 ┘
```

### Important: AB ≠ BA (Not Commutative!)

```
AB ≠ BA   in general!

Example:
A = [[1,2],[3,4]]    B = [[0,1],[1,0]]

AB = [[0+2, 1+0], [0+4, 3+0]] = [[2,1],[4,3]]
BA = [[0+3, 0+4], [1+0, 2+0]] = [[3,4],[1,2]]

AB ≠ BA  ✓ confirmed
```

### Python

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape 2×3

B = np.array([[ 7,  8],
              [ 9, 10],
              [11, 12]])       # shape 3×2

# Matrix multiplication
AB = A @ B          # @ is the matrix multiply operator
print("AB =\n", AB)
# Expected:
# [[ 58  64]
#  [139 154]]

# Verify one element manually
print("\nManual check (0,0):", 1*7 + 2*9 + 3*11)  # 58
print("Manual check (1,1):", 4*8 + 5*10 + 6*12)   # 154

# np.dot is equivalent
print("\nnp.dot(A, B) =\n", np.dot(A, B))

# Demonstrate non-commutativity
A2 = np.array([[1, 2], [3, 4]])
B2 = np.array([[0, 1], [1, 0]])
print("\nA2 @ B2 =\n", A2 @ B2)
print("B2 @ A2 =\n", B2 @ A2)
print("Are they equal?", np.allclose(A2 @ B2, B2 @ A2))  # False
```

---

## 8. Transpose of a Matrix

### What Is Transpose?

**Flip rows and columns.** The element at (i, j) moves to (j, i).

```
    ┌ 1  2  3 ┐               ┌ 1  4 ┐
A = │ 4  5  6 │   →   Aᵀ =   │ 2  5 │
    └           ┘              └ 3  6 ┘

A has shape 2×3.   Aᵀ has shape 3×2.
```

Step by step:
- Row 0 of A → Column 0 of Aᵀ: [1, 2, 3] becomes the first column
- Row 1 of A → Column 1 of Aᵀ: [4, 5, 6] becomes the second column

### Key Properties

```
(Aᵀ)ᵀ = A            Transpose of transpose = original
(A+B)ᵀ = Aᵀ + Bᵀ    Distributive
(AB)ᵀ = BᵀAᵀ         Order REVERSES!
```

### Symmetric Matrix

A matrix where **A = Aᵀ**. Only possible for square matrices.

```
    ┌ 1  2  3 ┐
S = │ 2  5  4 │    S is symmetric because S[i][j] = S[j][i]
    └ 3  4  9 ┘

S[0][1] = 2 = S[1][0]  ✓
S[0][2] = 3 = S[2][0]  ✓
S[1][2] = 4 = S[2][1]  ✓
```

Covariance matrices (used in PCA, statistics) are always symmetric.

### Python

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

AT = A.T
print("A shape:", A.shape)    # (2, 3)
print("Aᵀ shape:", AT.shape)  # (3, 2)
print("Aᵀ =\n", AT)

# Property: (AB)ᵀ = BᵀAᵀ
B = np.array([[7, 8], [9, 10], [11, 12]])
lhs = (A @ B).T
rhs = B.T @ A.T
print("\n(AB)ᵀ = BᵀAᵀ ?", np.allclose(lhs, rhs))  # True

# Symmetric matrix check
S = np.array([[1, 2, 3],
              [2, 5, 4],
              [3, 4, 9]])
print("\nS is symmetric:", np.allclose(S, S.T))  # True

# Create a symmetric matrix from any matrix: A + Aᵀ is always symmetric
M = np.array([[1, 2], [3, 4]])
sym = M + M.T
print("\nM + Mᵀ =\n", sym)
print("Symmetric?", np.allclose(sym, sym.T))  # True
```

---

## 9. Special Matrices

### 9.1 Identity Matrix (I)

The **multiplicative identity** — like the number 1 for matrices.

```
       ┌ 1  0  0 ┐
I₃ =   │ 0  1  0 │    1s on main diagonal, 0s everywhere else
       └ 0  0  1 ┘

AI = A    IA = A    for any matrix A
```

### 9.2 Zero Matrix

All elements are 0.

```
       ┌ 0  0 ┐
O₂ₓ₂ = │ 0  0 │
       └      ┘
```

### 9.3 Diagonal Matrix

Non-zero values only on the main diagonal.

```
    ┌ 3  0  0 ┐
D = │ 0 -1  0 │    Diagonal elements: {3, -1, 5}
    └ 0  0  5 ┘
```

Multiplying by a diagonal matrix just **scales each dimension**:

```
    ┌ 3  0 ┐   ┌ x ┐   ┌ 3x ┐
D × │      │ × │   │ = │    │
    └ 0  5 ┘   └ y ┘   └ 5y ┘
```

### 9.4 Upper / Lower Triangular Matrix

```
Upper triangular:           Lower triangular:
┌ 1  2  3 ┐               ┌ 1  0  0 ┐
│ 0  4  5 │               │ 2  4  0 │
└ 0  0  6 ┘               └ 3  5  6 ┘

Zeros below diagonal.      Zeros above diagonal.
```

### Python

```python
import numpy as np

# Identity matrix
I3 = np.eye(3)
print("I₃ =\n", I3)

# Verify AI = A
A = np.array([[1.0, 2], [3, 4]])
print("A @ I =\n", A @ np.eye(2))   # equals A

# Diagonal matrix
D = np.diag([3, -1, 5])
print("\nDiagonal matrix:\n", D)

# Extract diagonal from a matrix
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nDiagonal of M:", np.diag(M))   # [1, 5, 9]

# Triangular matrices
print("\nUpper triangular:\n", np.triu(M))
print("Lower triangular:\n", np.tril(M))

# Zero matrix
Z = np.zeros((3, 3))
print("\nZero matrix:\n", Z)
```

---

## 10. Determinants — Does the Matrix Stretch or Flip?

### What Is a Determinant?

The **determinant** is a single number that tells us:
1. Whether the matrix is invertible (det ≠ 0 → invertible)
2. How much the matrix **scales areas** (2D) or **volumes** (3D)
3. Whether the transformation **flips orientation** (negative determinant)

**Notation:** det(A) or |A|

### 2×2 Determinant — The Formula

```
    ┌ a  b ┐
A = │      │     det(A) = ad - bc
    └ c  d ┘
```

**Think of it as:** (top-left × bottom-right) − (top-right × bottom-left)

```
         ┌ 3  2 ┐
Example: │      │
         └ 1  4 ┘

det = (3 × 4) − (2 × 1)
    =    12   −    2
    =   10
```

Since det = 10 ≠ 0, this matrix is **invertible**. ✓

```
         ┌ 2  4 ┐
Example: │      │
         └ 1  2 ┘

det = (2 × 2) − (4 × 1)
    =    4   −    4
    =    0
```

det = 0 means rows are **linearly dependent** (row 2 = 0.5 × row 1). Matrix is **not invertible** (singular). ✗

### Geometric Meaning in 2D

The determinant equals the **area of the parallelogram** formed by the two column vectors.

```
Columns of A = [[3,2],[1,4]] are the two vectors:
  v₁ = (3, 1) and v₂ = (2, 4)

Area of parallelogram spanned by v₁ and v₂ = |det(A)| = 10
```

If you apply matrix A to a unit square, it becomes a parallelogram with area = |det(A)|.

### Python

```python
import numpy as np

# 2×2 determinant
A = np.array([[3, 2],
              [1, 4]])

det_manual = A[0,0]*A[1,1] - A[0,1]*A[1,0]
det_numpy  = np.linalg.det(A)

print("Manual det:", det_manual)   # 10
print("NumPy det:", det_numpy)     # 10.0

# Singular matrix (det = 0)
S = np.array([[2, 4],
              [1, 2]])
print("\nSingular matrix det:", np.linalg.det(S))  # ~0.0

# Visualize the geometric meaning
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

def plot_parallelogram(ax, A, title):
    v1 = A[:, 0]  # first column
    v2 = A[:, 1]  # second column
    origin = np.zeros(2)
    corners = np.array([origin, v1, v1+v2, v2, origin])
    ax.fill(corners[:, 0], corners[:, 1], alpha=0.3, color='blue')
    ax.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, color='red', label='v₁')
    ax.quiver(*origin, *v2, angles='xy', scale_units='xy', scale=1, color='green', label='v₂')
    ax.set_xlim(-1, 7); ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.legend(); ax.grid(True)
    ax.set_title(f"{title}\ndet = {np.linalg.det(A):.1f}")

plot_parallelogram(axes[0], np.eye(2), "Identity (unit square)")
plot_parallelogram(axes[1], A, "Transformed (area = det)")

plt.tight_layout()
plt.savefig("determinant_geometry.png", dpi=100)
plt.show()
```

### Properties of Determinants

```
det(AB) = det(A) × det(B)     Product rule
det(Aᵀ) = det(A)              Transpose same determinant
det(cA) = cⁿ det(A)           Scalar multiplication (n = matrix size)
det(A⁻¹) = 1/det(A)           Inverse
det(I) = 1                     Identity
```

Swapping two rows **negates** the determinant. Multiplying a row by k **multiplies** the determinant by k.

---

## 11. Determinant of a 3×3 Matrix (Cofactor Expansion)

### The Method: Expand Along First Row

```
    ┌ a  b  c ┐
A = │ d  e  f │
    └ g  h  i ┘

det(A) = a × M₁₁ − b × M₁₂ + c × M₁₃

Where Mᵢⱼ is the 2×2 determinant after removing row i and column j.
```

**Signs follow the checkerboard pattern:**
```
┌ +  −  + ┐
│ −  +  − │
└ +  −  + ┘
```

### Worked Example

```
    ┌ 1   2   3 ┐
A = │ 0   4   5 │
    └ 1   0   6 ┘
```

**Step 1:** Expand along row 0.

```
det(A) = 1 × |4  5| − 2 × |0  5| + 3 × |0  4|
             |0  6|         |1  6|         |1  0|
```

**Step 2:** Compute each 2×2 determinant.

```
|4  5|  = (4×6) − (5×0) = 24 − 0  =  24
|0  6|

|0  5|  = (0×6) − (5×1) =  0 − 5  = −5
|1  6|

|0  4|  = (0×0) − (4×1) =  0 − 4  = −4
|1  0|
```

**Step 3:** Combine with signs and elements.

```
det(A) = 1×(24)  −  2×(−5)  +  3×(−4)
       =   24    −   (−10)  +   (−12)
       =   24    +    10    −    12
       =   22
```

### Python

```python
import numpy as np

A = np.array([[1, 2, 3],
              [0, 4, 5],
              [1, 0, 6]])

# NumPy
det = np.linalg.det(A)
print("det(A) =", round(det))   # 22

# Manual cofactor expansion (showing the work)
a, b, c = A[0]
M11 = A[1,1]*A[2,2] - A[1,2]*A[2,1]   # det of minor removing row0, col0
M12 = A[1,0]*A[2,2] - A[1,2]*A[2,0]   # det of minor removing row0, col1
M13 = A[1,0]*A[2,1] - A[1,1]*A[2,0]   # det of minor removing row0, col2

det_manual = a*M11 - b*M12 + c*M13
print("Manual expansion:", det_manual)    # 22

print(f"\nBreakdown:")
print(f"  1 × {M11} = {1*M11}")          #  1 × 24 = 24
print(f"- 2 × {M12} = {-2*M12}")         # -2 × -5 = 10
print(f"+ 3 × {M13} = {3*M13}")          #  3 × -4 = -12
print(f"  Total = {det_manual}")          # 22

# Row operations effect on determinant
A_swapped = A[[1, 0, 2]]               # swap rows 0 and 1
print("\nSwap two rows → det negates:")
print("Original det:", round(np.linalg.det(A)))          # 22
print("After swap:", round(np.linalg.det(A_swapped)))    # -22
```

---

## 12. Trace of a Matrix

### Definition

The **trace** is the sum of all **diagonal elements** (only defined for square matrices).

```
    ┌ 3   2   1 ┐
A = │ 0  -1   4 │     tr(A) = 3 + (-1) + 5 = 7
    └ 2   3   5 ┘
                                ↑         ↑  ↑
                             A[0,0]  A[1,1]  A[2,2]
```

### Key Property

The trace equals the **sum of eigenvalues**. This makes it a quick sanity check when computing eigenvalues.

```
tr(A) = λ₁ + λ₂ + ... + λₙ
```

Also:
```
tr(AB) = tr(BA)         even though AB ≠ BA
tr(A + B) = tr(A) + tr(B)
tr(Aᵀ) = tr(A)
```

### Python

```python
import numpy as np

A = np.array([[3,  2, 1],
              [0, -1, 4],
              [2,  3, 5]])

trace = np.trace(A)
print("Trace:", trace)    # 7

# Verify: trace = sum of eigenvalues
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues:", eigenvalues)
print("Sum of eigenvalues:", sum(eigenvalues).real)  # ≈ 7.0

# tr(AB) = tr(BA)
B = np.random.randint(1, 5, (3, 3)).astype(float)
print("\ntr(AB):", np.trace(A @ B))
print("tr(BA):", np.trace(B @ A))   # same value
```

---

## 13. Matrix Rank — How Much Information?

### What Is Rank?

The **rank** of a matrix = the number of **linearly independent rows** (or columns — they give the same answer).

Intuitively: how many dimensions worth of information does the matrix actually contain?

```
    ┌ 1  2  3 ┐
A = │ 2  4  6 │    Row 2 = 2 × Row 1. Row 3 = 3 × Row 1.
    └ 3  6  9 ┘    All rows carry the same information!

rank(A) = 1   (only 1 truly independent row)
```

```
    ┌ 1  0  0 ┐
B = │ 0  1  0 │    All 3 rows are independent.
    └ 0  0  1 ┘    This is the identity matrix.

rank(B) = 3   (full rank)
```

### Why Rank Matters

- **Full rank** (rank = min(m,n)): matrix is "as informative as possible"
- **Rank deficient**: some rows/columns are redundant → matrix is singular → no unique solution to Ax=b
- In ML: **feature rank** tells you if your features contain redundant information (multicollinearity)

### Rank via Row Reduction (Gaussian Elimination)

```
    ┌ 1   2   1 ┐
A = │ 2   4   3 │
    └ 0   0   2 ┘

Step 1: R2 = R2 - 2×R1
    ┌ 1   2   1 ┐
    │ 0   0   1 │
    └ 0   0   2 ┘

Step 2: R3 = R3 - 2×R2
    ┌ 1   2   1 ┐
    │ 0   0   1 │
    └ 0   0   0 ┘

Non-zero rows: 2 → rank(A) = 2
```

### Python

```python
import numpy as np

# Rank-deficient matrix
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9]])
print("Rank of A (redundant rows):", np.linalg.matrix_rank(A))   # 1

# Full rank matrix
B = np.eye(3)
print("Rank of I₃:", np.linalg.matrix_rank(B))   # 3

# Rank via SVD (what numpy uses internally)
C = np.array([[1, 2, 1],
              [2, 4, 3],
              [0, 0, 2]])
print("Rank of C:", np.linalg.matrix_rank(C))    # 2 (as shown above)

# Check: rank ≤ min(rows, cols)
D = np.random.rand(5, 3)   # 5×3 random matrix
print(f"\n5×3 random matrix rank: {np.linalg.matrix_rank(D)}")  # likely 3

# Relationship: rank = number of non-zero singular values
_, s, _ = np.linalg.svd(C)
print("Singular values of C:", s)
print("Non-zero singular values:", np.sum(s > 1e-10))   # 2
```

---

## 14. Matrix Inverse — Undoing a Transformation

### What Is the Inverse?

For a square matrix A, its **inverse A⁻¹** satisfies:

```
A × A⁻¹ = I    and    A⁻¹ × A = I
```

Think of it as: if A transforms a vector, A⁻¹ **undoes** that transformation.

**Exists only when:** det(A) ≠ 0 (A is full rank, non-singular).

### 2×2 Inverse Formula

```
    ┌ a  b ┐             1     ┌  d  -b ┐
A = │      │    A⁻¹ = ——————— │        │
    └ c  d ┘           ad-bc   └ -c   a ┘
```

**Steps:**
1. Compute det(A) = ad − bc
2. Swap a and d
3. Negate b and c
4. Divide every element by det(A)

### Worked Example

```
    ┌ 4  7 ┐
A = │      │
    └ 2  6 ┘

Step 1: det(A) = (4×6) − (7×2) = 24 − 14 = 10

Step 2-4:
           1    ┌  6  -7 ┐   ┌  0.6  -0.7 ┐
A⁻¹  =  ———— × │        │ = │             │
          10    └ -2   4 ┘   └ -0.2   0.4 ┘
```

**Verify:** A × A⁻¹ should equal I.

```
┌ 4  7 ┐   ┌  0.6  -0.7 ┐
│      │ × │             │
└ 2  6 ┘   └ -0.2   0.4 ┘

(0,0): 4×0.6  + 7×(-0.2)  = 2.4 − 1.4 = 1.0  ✓
(0,1): 4×(-0.7) + 7×0.4   = -2.8 + 2.8 = 0.0  ✓
(1,0): 2×0.6  + 6×(-0.2)  = 1.2 − 1.2 = 0.0  ✓
(1,1): 2×(-0.7) + 6×0.4   = -1.4 + 2.4 = 1.0  ✓

Result = I  ✓
```

### Python

```python
import numpy as np

A = np.array([[4.0, 7],
              [2.0, 6]])

det = np.linalg.det(A)
print("det(A) =", det)    # 10.0

# Manual formula for 2×2
a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
A_inv_manual = (1/det) * np.array([[ d, -b],
                                    [-c,  a]])
print("\nManual A⁻¹:\n", A_inv_manual)

# NumPy inverse
A_inv = np.linalg.inv(A)
print("NumPy A⁻¹:\n", A_inv)

# Verify A × A⁻¹ = I
product = A @ A_inv
print("\nA × A⁻¹ =\n", np.round(product, 10))   # Should be I

# Singular matrix — inverse doesn't exist
S = np.array([[2.0, 4],
              [1.0, 2]])
print("\ndet(S) =", np.linalg.det(S))   # ~0
try:
    np.linalg.inv(S)
except np.linalg.LinAlgError as e:
    print("Error:", e)   # Singular matrix

# Pseudoinverse for non-square or singular matrices
A_pinv = np.linalg.pinv(S)
print("Pseudoinverse (pinv):\n", A_pinv)
```

---

## 15. Solving a Linear System Ax = b

### The Problem

Given a matrix A and vector b, find vector x such that:

```
Ax = b
```

In ML: this is what happens in **linear regression** — find weights x (called β) such that Xβ ≈ y.

### Method 1: Direct Inverse (when det ≠ 0)

```
Ax = b
A⁻¹(Ax) = A⁻¹b
Ix = A⁻¹b
x = A⁻¹b
```

### Worked Example

```
System of equations:
2x + y = 5
x + 3y = 10

Matrix form:
┌ 2  1 ┐ ┌ x ┐   ┌  5 ┐
│      │ │   │ = │    │
└ 1  3 ┘ └ y ┘   └ 10 ┘

det(A) = (2×3) − (1×1) = 6 − 1 = 5

A⁻¹ = (1/5) × ┌  3  -1 ┐ = ┌  0.6  -0.2 ┐
               └ -1   2 ┘   └ -0.2   0.4 ┘

x = A⁻¹ × b = ┌  0.6  -0.2 ┐ × ┌  5 ┐
               └ -0.2   0.4 ┘   └ 10 ┘

x[0] = 0.6×5 + (-0.2)×10 = 3 − 2 = 1
x[1] = (-0.2)×5 + 0.4×10 = -1 + 4 = 3

Solution: x=1, y=3

Verify: 2(1) + (3) = 5 ✓   and   (1) + 3(3) = 10 ✓
```

### Method 2: numpy.linalg.solve (More Numerically Stable)

Use `np.linalg.solve` instead of computing the inverse explicitly — it uses LU decomposition internally and is faster and more stable.

### Python

```python
import numpy as np

# System: 2x + y = 5,  x + 3y = 10
A = np.array([[2.0, 1],
              [1.0, 3]])
b = np.array([5.0, 10.0])

# Method 1: via inverse (illustrative, not recommended for large systems)
x_via_inv = np.linalg.inv(A) @ b
print("x via inverse:", x_via_inv)     # [1. 3.]

# Method 2: solve (preferred — numerically stable, no explicit inverse)
x = np.linalg.solve(A, b)
print("x via solve:", x)               # [1. 3.]

# Verify
print("Ax =", A @ x)   # should be [5, 10]
print("b  =", b)

# Linear Regression example — Normal Equation: β = (XᵀX)⁻¹ Xᵀy
# X is design matrix (n_samples × n_features), y is target vector
np.random.seed(42)
n = 50
X_raw = np.random.rand(n, 2)
X = np.hstack([np.ones((n, 1)), X_raw])   # add intercept column
true_beta = np.array([2.0, 3.0, -1.0])
y = X @ true_beta + 0.1 * np.random.randn(n)

# Normal equation: β = (XᵀX)⁻¹ Xᵀy
XtX = X.T @ X
Xty = X.T @ y
beta_hat = np.linalg.solve(XtX, Xty)   # solve instead of inv(XtX) @ Xty
print("\nEstimated β:", np.round(beta_hat, 3))
print("True β:     ", true_beta)
```

---

## 16. Putting It All Together — A Full ML Story

### Linear Regression from First Principles

This example uses every concept from this workbook in one end-to-end flow.

```
Problem: Predict house price from area (m²) and number of rooms.

Dataset:
Area  Rooms  Price($k)
 50     2       150
 80     3       200
120     4       300
 60     2       170
100     3       260
```

**Step 1: Build the matrix X and vector y**
```
X (design matrix with intercept):         y (targets):
┌ 1   50   2 ┐                            ┌ 150 ┐
│ 1   80   3 │                            │ 200 │
│ 1  120   4 │                            │ 300 │
│ 1   60   2 │                            │ 170 │
└ 1  100   3 ┘                            └ 260 ┘
```

**Step 2: Normal Equation**
```
β = (XᵀX)⁻¹ Xᵀy
```

**Step 3: Compute XᵀX** — a 3×3 symmetric matrix (always symmetric!)

**Step 4: Compute Xᵀy** — a 3D vector

**Step 5: Solve for β** using `np.linalg.solve`

### Python

```python
import numpy as np

# Data
X_raw = np.array([
    [ 50, 2],
    [ 80, 3],
    [120, 4],
    [ 60, 2],
    [100, 3]
], dtype=float)

y = np.array([150, 200, 300, 170, 260], dtype=float)

# Step 1: Add intercept column (column of 1s)
X = np.hstack([np.ones((5, 1)), X_raw])
print("Design matrix X:\n", X)
print("Shape:", X.shape)   # (5, 3)

# Step 2: Compute XᵀX
XtX = X.T @ X
print("\nXᵀX (3×3 symmetric matrix):\n", XtX)
print("XᵀX is symmetric:", np.allclose(XtX, XtX.T))   # True

# Step 3: Compute Xᵀy
Xty = X.T @ y
print("\nXᵀy:", Xty)

# Step 4: Check determinant — is XᵀX invertible?
det = np.linalg.det(XtX)
print("\ndet(XᵀX):", round(det, 2))
print("Invertible:", det != 0)

# Step 5: Solve normal equation
beta = np.linalg.solve(XtX, Xty)
print("\nEstimated coefficients β:", np.round(beta, 4))
print("  Intercept:", round(beta[0], 2))
print("  Area coeff:", round(beta[1], 2))
print("  Rooms coeff:", round(beta[2], 2))

# Step 6: Make predictions
y_pred = X @ beta
print("\nPredictions vs Actual:")
for actual, pred in zip(y, y_pred):
    print(f"  Actual: {actual:6.1f}  Predicted: {pred:6.1f}  Error: {actual-pred:+.1f}")

# Step 7: Compute R² score
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"\nR² score: {r_squared:.4f}")   # close to 1 = good fit

# === CONCEPTS USED IN THIS EXAMPLE ===
# ✓ Vectors — y is a vector, beta is a vector
# ✓ Matrix construction — X is a matrix
# ✓ Matrix transpose — X.T
# ✓ Matrix multiplication — X.T @ X, X.T @ y, X @ beta
# ✓ Symmetric matrix — XᵀX is always symmetric
# ✓ Determinant — checked if XᵀX is invertible
# ✓ Matrix inverse concept — (XᵀX)⁻¹ in normal equation
# ✓ Linear system — solve β = (XᵀX)⁻¹ Xᵀy via np.linalg.solve
# ✓ Vector norm — used internally in least-squares minimization
```

---

## Quick Reference Card

### Notation

| Symbol | Meaning |
|--------|---------|
| `v` | vector |
| `A`, `B` | matrices |
| `Aᵀ` | transpose of A |
| `A⁻¹` | inverse of A |
| `det(A)` or `\|A\|` | determinant |
| `tr(A)` | trace (sum of diagonal) |
| `rank(A)` | number of independent rows |
| `\|\|v\|\|` | norm (length) of vector |
| `I` | identity matrix |
| `λ` | eigenvalue |

### When to Use What

| Question | Tool |
|----------|------|
| Can I invert this matrix? | Check `det(A) ≠ 0` |
| Are my features redundant? | Check `rank(A)` |
| How much does A stretch space? | Compute `det(A)` |
| What direction is dominant? | Eigenvalues/eigenvectors |
| Solve `Ax = b` | `np.linalg.solve(A, b)` |
| Fit a linear model | Normal equation `β = (XᵀX)⁻¹Xᵀy` |
| Measure vector similarity | Cosine similarity = dot product / (norms) |
| Regularize model weights | L1 or L2 norm penalty |

### Shape Rules

| Operation | Shape rule | Result |
|-----------|-----------|--------|
| `A + B` | A and B same shape | same shape |
| `A @ B` | A is m×n, B is n×p | m×p |
| `Aᵀ` | A is m×n | n×m |
| `A⁻¹` | A must be square n×n, det≠0 | n×n |
| `det(A)` | A must be square | scalar |
| `tr(A)` | A must be square | scalar |

---

*This workbook is based on the "Machine Learning Mathematics" series by Dr. Rajendra (Demystify with Ajay).*
