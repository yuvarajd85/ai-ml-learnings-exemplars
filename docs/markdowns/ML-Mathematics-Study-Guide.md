# Machine Learning Mathematics: Complete Study Guide

> Based on the "Demystify with Ajay" series by Dr. Rajendra.
> Covers all foundational linear algebra topics essential for Machine Learning.

---

## Table of Contents

1. [Vectors & Vector Spaces](#1-vectors--vector-spaces)
2. [Matrix Fundamentals](#2-matrix-fundamentals)
3. [Determinants](#3-determinants)
4. [Matrix Inverse](#4-matrix-inverse)
5. [Orthogonality & Orthogonal Matrices](#5-orthogonality--orthogonal-matrices)
6. [QR Decomposition](#6-qr-decomposition)
7. [Eigenvalues & Eigenvectors](#7-eigenvalues--eigenvectors)
8. [Matrix Diagonalization](#8-matrix-diagonalization)
9. [Symmetric Matrices](#9-symmetric-matrices)
10. [Singular Matrices & Rank](#10-singular-matrices--rank)
11. [Quadratic Forms](#11-quadratic-forms)
12. [Matrix Definiteness](#12-matrix-definiteness)
13. [Generalized Eigendecomposition](#13-generalized-eigendecomposition)
14. [Principal Component Analysis (PCA)](#14-principal-component-analysis-pca)
15. [Covariance & Correlation Matrices](#15-covariance--correlation-matrices)
16. [Geometric Transformations](#16-geometric-transformations)
17. [Image Filtering & Convolution](#17-image-filtering--convolution)
18. [Numerical Stability & Condition Number](#18-numerical-stability--condition-number)

---

## 1. Vectors & Vector Spaces

### Concept

A **vector** is an ordered list of numbers representing a point or direction in space. A **vector space** is the collection of all vectors that can be formed by linear combinations of a set of basis vectors.

**Key subspaces for a matrix A:**
- **Column space (range):** all vectors reachable as `Ax` — the span of A's columns
- **Null space (kernel):** all vectors `x` such that `Ax = 0`

### Core Formulas

**Dot product:** `a · b = Σ aᵢbᵢ = ||a|| ||b|| cos(θ)`

**Vector norms:**
- L1 norm: `||x||₁ = Σ |xᵢ|`
- L2 norm: `||x||₂ = √(Σ xᵢ²)`

**Frobenius norm (matrix):** `||A||_F = √(Σᵢ Σⱼ aᵢⱼ²)`

**Linear combination:** `v = c₁v₁ + c₂v₂ + ... + cₖvₖ`

**Span:** All linear combinations of a set of vectors.

**Linear independence:** No vector in the set can be written as a linear combination of the others.

### Geometric Meaning

- The **dot product** measures similarity in direction; it is zero when vectors are perpendicular.
- The **column space** is the set of all outputs a matrix can produce — a geometric "reach."
- A vector lies in the column space of A if and only if augmenting A with that vector does **not increase the rank**.

### ML Application

- Feature vectors in ML are points in a high-dimensional vector space.
- **Column space membership test** is used to determine if a target is achievable by a linear model.
- **Norms** are used in regularization (L1 → Lasso, L2 → Ridge).

### Python Exemplar

```python
import numpy as np
from scipy.linalg import null_space

# ------ Vectors and norms ------
a = np.array([3.0, 4.0])
b = np.array([1.0, 0.0])

print("L1 norm:", np.linalg.norm(a, 1))          # 7.0
print("L2 norm:", np.linalg.norm(a, 2))          # 5.0
print("Dot product:", np.dot(a, b))              # 3.0
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("Angle (deg):", np.degrees(np.arccos(cos_theta)))  # 53.13

# ------ Column space membership ------
A = np.array([[1, 3],
              [3, 1],
              [5, 1],
              [1, 0]], dtype=float)   # 4x2 matrix; column space is 2D in R^4

v1 = 2*A[:, 0] + 1*A[:, 1]          # guaranteed in column space
v2 = np.array([0, 0, 0, 1], dtype=float)  # test vector

rank_A = np.linalg.matrix_rank(A)

for label, v in [("v1", v1), ("v2", v2)]:
    augmented = np.column_stack([A, v])
    rank_aug = np.linalg.matrix_rank(augmented)
    if rank_aug == rank_A:
        print(f"{label} IS in the column space of A")
    else:
        print(f"{label} is NOT in the column space of A")

# ------ Frobenius norm ------
M = np.array([[1, 2], [3, 4]], dtype=float)
print("Frobenius norm:", np.linalg.norm(M, 'fro'))   # sqrt(1+4+9+16) = sqrt(30)
```

---

## 2. Matrix Fundamentals

### Concept

A **matrix** is a rectangular array of numbers that represents a linear transformation. An `m × n` matrix maps vectors from `Rⁿ` to `Rᵐ`.

**Key properties:**
- **Transpose:** `(A^T)ᵢⱼ = Aⱼᵢ` — flip rows and columns
- **Trace:** `tr(A) = Σᵢ aᵢᵢ` — sum of diagonal elements
- **Rank:** number of linearly independent rows (= linearly independent columns)

### Core Formulas

**Matrix product:** `(AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ` (row-by-column dot products)

**Key identities:**
- `(AB)^T = B^T A^T`
- `tr(AB) = tr(BA)`
- `rank(A) = rank(A^T)` — row rank equals column rank
- `rank(AB) ≤ min(rank(A), rank(B))`

### Geometric Meaning

- Matrix multiplication = composing linear transformations.
- Rank measures the "true dimensionality" of the output space.
- The trace equals the sum of eigenvalues (proven via characteristic polynomial).

### ML Application

- Data matrices: rows = samples, columns = features.
- Covariance matrix: `C = (1/n-1) X̃^T X̃` — always square and symmetric.
- Gram matrix: `G = X^T X` — used in kernel methods and PCA.

### Python Exemplar

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=float)   # 2x3

B = np.array([[7, 8],
              [9, 10],
              [11, 12]], dtype=float)   # 3x2

# Matrix multiplication: (2x3) @ (3x2) = (2x2)
C = A @ B
print("A @ B =\n", C)

# Transpose
print("A^T shape:", A.T.shape)         # (3, 2)
print("(AB)^T == B^T A^T:", np.allclose((A@B).T, B.T @ A.T))

# Trace (only for square matrix)
S = np.array([[4, 2, 0],
              [2, 3, 1],
              [0, 1, 5]], dtype=float)
print("Trace:", np.trace(S))           # 4+3+5 = 12

# Rank
print("Rank of A:", np.linalg.matrix_rank(A))   # 2 (full row rank)

# Rank-deficient matrix
D = np.array([[1, 2, 3],
              [2, 4, 6],    # row 2 = 2 * row 1
              [0, 1, 0]], dtype=float)
print("Rank of D (deficient):", np.linalg.matrix_rank(D))   # 2, not 3
```

---

## 3. Determinants

### Concept

The **determinant** is a scalar value that encodes how a square matrix scales area/volume under transformation. It reveals critical structural properties of the matrix.

### Core Formula

**2×2 matrix:**
```
det([[a, b], [c, d]]) = ad - bc
```

**General (Laplace expansion along row 1):**
```
det(A) = Σⱼ (-1)^(1+j) a₁ⱼ det(Mᵢⱼ)
```
where `Mᵢⱼ` is the minor (submatrix with row i and col j removed).

**Properties:**
- `det(AB) = det(A) · det(B)`
- `det(A^T) = det(A)`
- `det(A⁻¹) = 1 / det(A)`
- `det(A) = Π λᵢ` (product of all eigenvalues)

### Geometric Meaning

- `|det(A)|` = scaling factor of area (2D) or volume (nD) under transformation A.
- `det(A) > 0`: orientation preserved. `det(A) < 0`: orientation flipped.
- `det(A) = 0`: the transformation collapses at least one dimension — the matrix is **singular** (non-invertible).
- Rotation matrices have `det = 1` (pure rotation, no area change, no reflection).

### ML Application

- `det(A) = 0` → rank-deficient design matrix → linear regression has no unique solution.
- Product of eigenvalues = determinant → singular matrix has at least one zero eigenvalue.
- Gaussian distribution: `det(Σ)` appears in the normalization constant of the multivariate Gaussian PDF.

### Python Exemplar

```python
import numpy as np

# 2x2 determinant: ad - bc
A = np.array([[1, 4],
              [2, 7]], dtype=float)
a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
det_manual = a*d - b*c
det_numpy  = np.linalg.det(A)
print(f"Manual det: {det_manual}, NumPy det: {det_numpy:.4f}")  # both -1

# Determinant = 0 → singular
A_singular = np.array([[1, 4],
                        [2, 8]], dtype=float)  # row 2 = 2 * row 1
print("Singular det:", np.linalg.det(A_singular))   # ~0.0

# det(AB) = det(A) * det(B)
B = np.array([[3, 1],
              [2, 5]], dtype=float)
print("det(A)*det(B):", np.linalg.det(A) * np.linalg.det(B))
print("det(A@B)     :", np.linalg.det(A @ B))  # should match

# Rotation matrix: det = 1
theta = np.pi / 4  # 45 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
print("det(Rotation):", np.linalg.det(R))   # 1.0

# Eigenvalue product = determinant
S = np.array([[4, 2],
              [2, 3]], dtype=float)
evals, _ = np.linalg.eig(S)
print("Product of eigenvalues:", np.prod(evals))
print("det(S)               :", np.linalg.det(S))   # both should match

# Cofactor (adjugate) matrix inverse: A_inv = (1/det) * C^T
# For 2x2: A_inv = (1/det) * [[d, -b], [-c, a]]
det_A = np.linalg.det(A)
A_inv_manual = (1/det_A) * np.array([[ d, -b],
                                      [-c,  a]])
A_inv_numpy = np.linalg.inv(A)
print("Manual inverse:\n", A_inv_manual)
print("NumPy inverse:\n",  A_inv_numpy)
```

---

## 4. Matrix Inverse

### Concept

The **inverse** `A⁻¹` of a square matrix `A` reverses the transformation: `A⁻¹A = AA⁻¹ = I`. It exists **if and only if** `det(A) ≠ 0` (i.e., A is full rank / non-singular).

**Types of inverses:**
1. **Full inverse:** `A⁻¹ A = A A⁻¹ = I` — requires square + full rank.
2. **Left inverse:** `A_L⁻¹ A = I` — exists for tall matrices with full column rank: `A_L⁻¹ = (A^T A)⁻¹ A^T`
3. **Right inverse:** `A A_R⁻¹ = I` — exists for wide matrices with full row rank: `A_R⁻¹ = A^T (AA^T)⁻¹`
4. **Moore-Penrose pseudoinverse:** `A⁺` — works for any matrix shape and rank.

**Uniqueness:** An invertible matrix has exactly **one** inverse. (Proof: if B and C are both inverses of A, then `B = BI = B(AC) = (BA)C = IC = C`.)

### Core Formulas

**2×2 inverse:**
```
A = [[a, b], [c, d]]  →  A⁻¹ = 1/(ad-bc) * [[d, -b], [-c, a]]
```

**Left inverse (tall matrix, full column rank):**
```
A_L⁻¹ = (A^T A)⁻¹ A^T
```

**Right inverse (wide matrix, full row rank):**
```
A_R⁻¹ = A^T (A A^T)⁻¹
```

**Cofactor/Adjugate method (general):**
```
A⁻¹ = (1/det(A)) * adj(A)
adj(A)ᵢⱼ = (-1)^(i+j) det(Mⱼᵢ)   (note: transposed cofactors)
```

**Diagonal matrix inverse:**
```
D = diag(d₁, d₂, ..., dₙ)  →  D⁻¹ = diag(1/d₁, 1/d₂, ..., 1/dₙ)
```
(Fails if any dᵢ = 0.)

### Geometric Meaning

If `A` transforms points from shape `P` → shape `Q`, then `A⁻¹` is the transformation that maps `Q` back to `P`. A singular matrix "collapses" at least one dimension; that information is lost and cannot be recovered — hence no inverse.

### ML Application

- **Solving Ax = b:** `x = A⁻¹b` — parameter estimation in linear regression.
- **Normal equation:** `θ = (X^T X)⁻¹ X^T y` — ordinary least squares.
- **Pseudoinverse** powers least-squares regression when `X^T X` is rank-deficient.
- **Practical note:** Never compute `A⁻¹` directly in code if you only need to solve `Ax = b`; use `np.linalg.solve` (more stable).

### Python Exemplar

```python
import numpy as np
from numpy.linalg import inv, pinv, matrix_rank

# ---- Full 2x2 inverse (manual vs NumPy) ----
A = np.array([[1, 4],
              [2, 7]], dtype=float)
a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
det = a*d - b*c
A_inv_manual = (1/det) * np.array([[d, -b], [-c, a]])
A_inv_numpy  = inv(A)
print("Manual A_inv:\n", A_inv_manual)
print("NumPy  A_inv:\n", A_inv_numpy)
print("A @ A_inv ≈ I:", np.allclose(A @ A_inv_numpy, np.eye(2)))

# ---- Left inverse (tall matrix) ----
T = np.random.default_rng(42).standard_normal((8, 3))  # 8x3, full column rank
T_left_inv = inv(T.T @ T) @ T.T   # (3x8) left inverse
print("T_left_inv @ T ≈ I:", np.allclose(T_left_inv @ T, np.eye(3)))

# ---- Right inverse (wide matrix) ----
W = np.random.default_rng(7).standard_normal((3, 8))   # 3x8, full row rank
W_right_inv = W.T @ inv(W @ W.T)  # (8x3) right inverse
print("W @ W_right_inv ≈ I:", np.allclose(W @ W_right_inv, np.eye(3)))

# ---- Moore-Penrose Pseudoinverse ----
# Works for ANY shape or rank
for name, M in [("Square singular", np.array([[1,2],[2,4]], dtype=float)),
                ("Tall (5x3)",       T[:5, :]),
                ("Wide (3x5)",       W[:, :5])]:
    Mplus = pinv(M)
    print(f"\n{name}: shape {M.shape}, pinv shape {Mplus.shape}")
    # Verify: M pinv(M) M ≈ M  (Moore-Penrose condition 1)
    print("  M @ pinv(M) @ M ≈ M:", np.allclose(M @ Mplus @ M, M))

# ---- Diagonal matrix inverse ----
d_vals = np.array([2.0, 3.0, 4.0])
D = np.diag(d_vals)
D_inv = np.diag(1.0 / d_vals)
print("\nD @ D_inv ≈ I:", np.allclose(D @ D_inv, np.eye(3)))

# ---- Prefer solve over inv for Ax=b ----
b_vec = np.array([1.0, 2.0])
x_via_inv   = inv(A) @ b_vec
x_via_solve = np.linalg.solve(A, b_vec)   # more stable
print("\nx via inv  :", x_via_inv)
print("x via solve:", x_via_solve)

# ---- Cofactor/Adjugate (MCA) for 3x3 ----
def matrix_inverse_mca(A):
    n = A.shape[0]
    det = np.linalg.det(A)
    if abs(det) < 1e-12:
        raise ValueError("Matrix is singular")
    minors = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sub = np.delete(np.delete(A, i, axis=0), j, axis=1)
            minors[i, j] = np.linalg.det(sub)
    sign_grid = np.array([[(-1)**(i+j) for j in range(n)] for i in range(n)])
    cofactors = minors * sign_grid
    adjugate  = cofactors.T       # adjugate = transpose of cofactor matrix
    return adjugate / det

A3 = np.array([[2, 1, 3],
               [1, 4, 2],
               [0, 1, 5]], dtype=float)
A3_inv_mca   = matrix_inverse_mca(A3)
A3_inv_numpy = inv(A3)
print("\nMCA inverse matches NumPy:", np.allclose(A3_inv_mca, A3_inv_numpy))
```

---

## 5. Orthogonality & Orthogonal Matrices

### Concept

Two vectors are **orthogonal** (perpendicular) when their dot product is zero: `u · v = 0`.

An **orthogonal matrix** Q satisfies: `Q^T Q = I`, which means `Q⁻¹ = Q^T`.

Properties:
- Columns are mutually orthogonal unit vectors (orthonormal).
- Preserves lengths: `||Qx|| = ||x||` for all x.
- Preserves angles between vectors.
- `det(Q) = ±1` (rotation if +1, reflection if -1).

### Core Formula

```
Q^T Q = I   ↔   Q⁻¹ = Q^T
```

**Rotation matrix (2D, counterclockwise by θ):**
```
R(θ) = [[cos θ, -sin θ],
         [sin θ,  cos θ]]
```

**Gram-Schmidt orthogonalization:** given vectors {a₁, a₂, ..., aₖ}:
```
u₁ = a₁
u₂ = a₂ - (a₂·u₁/u₁·u₁) u₁
uₖ = aₖ - Σⱼ<ₖ (aₖ·uⱼ/uⱼ·uⱼ) uⱼ
eᵢ = uᵢ / ||uᵢ||   (normalize)
```

### Geometric Meaning

Orthogonal matrices are "rigid-body transformations" — they rotate (or reflect) without any stretching or squishing. The inverse is free to compute (`Q^T` instead of Gaussian elimination), making them numerically ideal.

### ML Application

- Rotation matrices appear in **PCA** (eigenvectors of covariance matrix form Q).
- `Q^T Q = I` makes `Q^(-1) = Q^T` — transpose is far cheaper and more stable than matrix inversion.
- Gram-Schmidt is the foundation of QR decomposition.
- **Signal processing:** orthogonal transforms (DFT, DCT, wavelets) are built on orthogonal matrices.

### Python Exemplar

```python
import numpy as np

# ---- Rotation matrix is orthogonal ----
theta = np.pi / 5   # 36 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print("R^T R:\n", np.round(R.T @ R, 10))          # identity
print("R^(-1) == R^T:", np.allclose(np.linalg.inv(R), R.T))
print("det(R) =", np.linalg.det(R))               # 1.0 (pure rotation)

# Rotation preserves length
x = np.array([3.0, 4.0])
Rx = R @ x
print("||x|| =", np.linalg.norm(x))               # 5.0
print("||Rx||=", np.linalg.norm(Rx))              # still 5.0

# ---- Gram-Schmidt orthogonalization ----
def gram_schmidt(A):
    Q = np.zeros_like(A, dtype=float)
    for i in range(A.shape[1]):
        v = A[:, i].astype(float)
        for j in range(i):
            v = v - np.dot(Q[:, j], v) * Q[:, j]  # project out previous directions
        Q[:, i] = v / np.linalg.norm(v)           # normalize
    return Q

A = np.array([[1, 2, 0],
              [0, 3, 4],
              [1, 1, 5]], dtype=float)

Q_gs = gram_schmidt(A)
print("\nGram-Schmidt Q:\n", np.round(Q_gs, 4))
print("Q^T Q ≈ I:", np.allclose(Q_gs.T @ Q_gs, np.eye(3)))

# ---- Pairwise orthogonality check ----
n = Q_gs.shape[1]
for i in range(n):
    for j in range(i+1, n):
        dot = np.dot(Q_gs[:, i], Q_gs[:, j])
        print(f"v{i}·v{j} = {dot:.2e}")   # ~0 for all pairs
```

---

## 6. QR Decomposition

### Concept

**QR decomposition** factors any matrix `A` into:
- `Q`: orthogonal matrix (columns = orthonormal basis for column space of A)
- `R`: upper triangular matrix (stores the structural information about A)

```
A = Q R
```

Gram-Schmidt builds Q from A's columns but "discards" information. That discarded information is captured in R via:
```
R = Q^T A
```
(since `Q^T Q = I` → `Q^T (QR) = R`)

**Economy QR:** Q is m×n and R is n×n when A is m×n with m > n.
**Full QR:** Q is m×m (complete orthogonal basis for Rᵐ).

### Core Formula

**QR-based matrix inverse:**
```
A⁻¹ = (QR)⁻¹ = R⁻¹ Q^T
```
R is upper triangular → `R⁻¹` via back substitution (stable and efficient). Q is orthogonal → `Q⁻¹ = Q^T` (free).

**Least squares via QR:**
```
A^T A x = A^T b  →  R x = Q^T b   (more stable than normal equations)
```

### Geometric Meaning

Q captures the "geometry" (orientation, rotation) of A's column space. R captures the "structure" (how much each vector contributes). QR peels apart direction from magnitude.

### ML Application

- **Numerically stable least-squares** regression (avoids squaring condition number that happens with normal equations `X^T X`).
- **Eigenvalue algorithms** (QR iteration — the engine behind `np.linalg.eig`).
- Provides stable path to `A⁻¹` avoiding numerical errors of direct Gaussian elimination.

### Python Exemplar

```python
import numpy as np
from scipy.linalg import solve_triangular

np.random.seed(42)
A = np.random.randn(6, 6)

# ---- QR decomposition ----
Q, R = np.linalg.qr(A)

print("Q shape:", Q.shape)              # (6,6)
print("R shape:", R.shape)              # (6,6)
print("Q^T Q ≈ I:", np.allclose(Q.T @ Q, np.eye(6)))
print("A ≈ QR:  ", np.allclose(A, Q @ R))

# Check R is upper triangular
print("R upper triangular (below diag ≈ 0):", np.allclose(np.tril(R, -1), 0))

# ---- QR-based matrix inverse: A^(-1) = R^(-1) Q^T ----
# R^(-1) via back substitution (stable)
R_inv = solve_triangular(R, np.eye(6))
A_inv_qr    = R_inv @ Q.T
A_inv_numpy = np.linalg.inv(A)
print("\nQR-based inv ≈ NumPy inv:", np.allclose(A_inv_qr, A_inv_numpy))

# ---- Economy QR for overdetermined system (Ax ≈ b, least squares) ----
m, n = 20, 4
A_tall = np.random.randn(m, n)
b = np.random.randn(m)

# Stable: QR then back substitution
Q2, R2 = np.linalg.qr(A_tall, mode='reduced')  # economy QR
Qtb = Q2.T @ b          # Q^T b
x_qr = solve_triangular(R2, Qtb)

# Compare with normal equations (less stable)
x_normal = np.linalg.solve(A_tall.T @ A_tall, A_tall.T @ b)

print("\nLeast-squares solutions match:", np.allclose(x_qr, x_normal, atol=1e-8))
print("Residual (QR method):", np.linalg.norm(A_tall @ x_qr - b))

# ---- Hilbert matrix: ill-conditioned, QR is more stable ----
from scipy.linalg import hilbert
H = hilbert(8)
print("\nHilbert 8x8 condition number:", np.linalg.cond(H))   # very large
Q_h, R_h = np.linalg.qr(H)
print("Q_H is orthogonal:", np.allclose(Q_h.T @ Q_h, np.eye(8), atol=1e-10))
```

---

## 7. Eigenvalues & Eigenvectors

### Concept

An **eigenvector** is a special non-zero vector `v` such that when a matrix `A` multiplies it, the result points in the **same direction** — only scaled by a scalar `λ` (the **eigenvalue**):

```
A v = λ v
```

The matrix rotates and scales most vectors. Eigenvectors are the exceptions — they only scale, not rotate. This is the defining property.

**Key facts:**
- Only square (m×m) matrices have eigenvalues/eigenvectors.
- An m×m matrix has exactly m eigenvalues (counting multiplicity).
- Eigenvectors are **not unique**: if v is an eigenvector, so is αv for any scalar α ≠ 0.

### Core Formulas

**Characteristic equation (to find eigenvalues):**
```
det(A - λI) = 0
```
This produces a degree-m polynomial in λ — the **characteristic polynomial**.

**Finding eigenvectors (for each λᵢ):**
```
v ∈ null space of (A - λᵢI)
i.e., solve: (A - λᵢI) v = 0
```

**Trace and determinant invariants:**
```
tr(A) = Σᵢ λᵢ        (sum of eigenvalues = trace)
det(A) = Πᵢ λᵢ       (product of eigenvalues = determinant)
```
These hold for any square matrix, regardless of symmetry or diagonalizability.

### Geometric Meaning

- **Eigenvectors** define the "natural axes" of the transformation — directions the matrix doesn't rotate.
- **Eigenvalues** tell how much stretching occurs along each axis.
  - `|λ| > 1`: stretch; `|λ| < 1`: shrink; `λ = -1`: flip
- The characteristic polynomial's roots reveal these stretching factors.

### ML Application

- **PCA**: eigenvectors of the covariance matrix = principal components; eigenvalues = variance explained.
- **Graph Laplacians**: eigenvectors used in spectral clustering.
- **PageRank**: stationary distribution found via dominant eigenvector.
- **System stability**: eigenvalue magnitudes determine whether dynamical systems converge.

### Python Exemplar

```python
import numpy as np
from scipy.linalg import null_space

# ---- Define matrix and compute eigenvalues ----
A = np.array([[1, 3],
              [3, 1]], dtype=float)

# Step 1: characteristic polynomial  det(A - λI) = 0
# det([[1-λ, 3],[3, 1-λ]]) = (1-λ)^2 - 9 = λ^2 - 2λ - 8 = (λ-4)(λ+2) = 0
# λ₁ = 4,  λ₂ = -2

evals, evecs = np.linalg.eig(A)
print("Eigenvalues:", evals)         # [4, -2] (or [-2, 4])
print("Eigenvectors (columns):\n", evecs)

# Step 2: verify Av = λv for each pair
for i in range(len(evals)):
    lam = evals[i]
    v   = evecs[:, i]
    Av  = A @ v
    lam_v = lam * v
    print(f"\nλ={lam:.1f}: Av={Av}, λv={lam_v}, match={np.allclose(Av, lam_v)}")

# Step 3: find eigenvector via null space of (A - λI)
for lam in evals:
    shifted = A - lam * np.eye(2)         # A - λI
    ns = null_space(shifted)              # null space = eigenvector direction
    v_ns = ns[:, 0]
    v_ns = v_ns / np.linalg.norm(v_ns)   # normalize
    print(f"\nλ={lam:.1f}: null space eigenvector = {v_ns}")

# Step 4: trace = sum of eigenvalues, det = product
print("\ntr(A)          =", np.trace(A))
print("sum(eigenvalues)=", np.sum(evals))

print("\ndet(A)            =", np.linalg.det(A))
print("product(eigenvalues)=", np.prod(evals))

# ---- Geometric illustration: eigenvectors don't rotate ----
rng = np.random.default_rng(0)
# Verify: Av is in same direction as v for eigenvectors
for i in range(2):
    v = evecs[:, i]
    Av = A @ v
    # angle between v and Av should be 0 or π
    cos_angle = np.dot(v, Av) / (np.linalg.norm(v) * np.linalg.norm(Av))
    print(f"Eigenvector {i}: cos(angle between v and Av) = {cos_angle:.4f}")
    # cos = ±1 means same or opposite direction — eigenvector property confirmed
```

---

## 8. Matrix Diagonalization

### Concept

**Diagonalization** rewrites a matrix in terms of its eigenvectors and eigenvalues, revealing its simplest form:

```
A = V Λ V⁻¹
```

where:
- `V`: matrix whose **columns are eigenvectors** of A
- `Λ`: **diagonal matrix** with eigenvalues on the diagonal
- `V⁻¹`: inverse of eigenvector matrix

Collecting all m eigen equations `Avᵢ = λᵢvᵢ` into one matrix equation:
```
AV = VΛ   →   A = VΛV⁻¹
```

**Conditions:** A is diagonalizable if and only if it has m linearly independent eigenvectors.

### Core Formulas

```
A = V Λ V⁻¹          (diagonalization / eigendecomposition)
Aᵏ = V Λᵏ V⁻¹        (efficient matrix power)
tr(A) = Σᵢ λᵢ = Σᵢ Aii  (both are equal)
det(A) = Πᵢ λᵢ
```

### Geometric Meaning

In the standard basis, A looks complex (entries everywhere). In the **eigenvector basis**, A is simply a diagonal scaling — scaling each axis independently by its eigenvalue. Diagonalization finds the coordinate system in which the transformation has its simplest form.

### ML Application

- **Efficient computation of Aᵏ**: used in Markov chains, recurrence relations, PageRank.
- **Matrix exponential** `e^A = V e^Λ V⁻¹`: used in differential equations, NLP attention.
- Foundation of **PCA**: diagonalizing the covariance matrix.
- **Dimensionality reduction**: keep only k largest eigenvalue/vector pairs.

### Python Exemplar

```python
import numpy as np

# ---- Matrix diagonalization ----
A = np.array([[4, 2],
              [1, 3]], dtype=float)

evals, V = np.linalg.eig(A)
Lambda = np.diag(evals)        # diagonal matrix of eigenvalues
V_inv  = np.linalg.inv(V)      # inverse of eigenvector matrix

# Reconstruct A from V, Λ, V⁻¹
A_reconstructed = V @ Lambda @ V_inv
print("A reconstructed from VΛV^(-1):\n", np.round(A_reconstructed, 6))
print("Matches original A:", np.allclose(A_reconstructed, A))

# Core identity: AV = VΛ
print("\nAV ≈ VΛ:", np.allclose(A @ V, V @ Lambda))

# ---- Trace and determinant invariants ----
print("\ntr(A)        :", np.trace(A))
print("sum(λ)       :", np.sum(evals))
print("det(A)       :", np.linalg.det(A))
print("product(λ)   :", np.prod(evals))

# ---- Efficient matrix powers via diagonalization ----
# A^k = V Λ^k V^(-1)
k = 10
A_power_direct = np.linalg.matrix_power(A, k)
Lambda_k = np.diag(evals**k)       # just raise diagonal to power k
A_power_eigen = V @ Lambda_k @ V_inv
print(f"\nA^{k} via direct power vs diagonalization match:",
      np.allclose(A_power_direct, A_power_eigen))

# ---- 3x3 example with verification ----
A3 = np.array([[2, 1, 0],
               [1, 3, 1],
               [0, 1, 2]], dtype=float)

evals3, V3 = np.linalg.eig(A3)
Lambda3 = np.diag(evals3)
V3_inv  = np.linalg.inv(V3)

print("\n3x3 eigenvalues:", np.round(evals3, 4))
print("A3 = V Λ V^(-1) :", np.allclose(A3, V3 @ Lambda3 @ V3_inv))
print("tr(A3) = sum(λ):", np.allclose(np.trace(A3), np.sum(evals3)))
print("det(A3) = prod(λ):", np.allclose(np.linalg.det(A3), np.prod(evals3)))
```

---

## 9. Symmetric Matrices

### Concept

A matrix is **symmetric** if `A = A^T` (equal to its transpose). Symmetric matrices have two remarkable properties that make them the "superheroes of linear algebra":

1. **All eigenvalues are real** (no complex numbers, even if the matrix has all-real entries in the non-symmetric case).
2. **Eigenvectors corresponding to distinct eigenvalues are orthogonal** — they are always perpendicular.

This leads to the **Spectral Theorem**: every real symmetric matrix has an **orthogonal diagonalization**:
```
A = V Λ V^T    (NOT V Λ V^(-1), because V^(-1) = V^T for symmetric A)
```

### Core Formulas

```
A = A^T                         (symmetry condition)
A = V Λ V^T                     (spectral theorem: orthogonal diagonalization)
V^T V = I  →  V^(-1) = V^T     (eigenvectors form an orthogonal matrix)
```

**Proof that distinct eigenvalues → orthogonal eigenvectors:**
Starting from `Av₁ = λ₁v₁`, left-multiply by `v₂^T`:
```
v₂^T A v₁ = λ₁ (v₂^T v₁)
```
Since A is symmetric, `A^T = A`:
```
v₁^T A v₂ = λ₂ (v₁^T v₂)  →  v₂^T A v₁ = λ₂ (v₂^T v₁)
```
Equating: `λ₁(v₂^T v₁) = λ₂(v₂^T v₁)` → `(λ₁ - λ₂)(v₂^T v₁) = 0`. Since `λ₁ ≠ λ₂`, we get `v₂^T v₁ = 0`. ∎

### Geometric Meaning

Symmetric matrices perform **pure scaling along orthogonal axes** — no rotational component. This is why their eigenvalues must be real: there's no "twisting" that would produce complex values. The transformation is geometrically simple: stretch or shrink, but never spin.

### ML Application

- **Covariance matrices** are always symmetric (by definition): `C = X^T X / (n-1)`.
- `np.linalg.eigh` is specialized for symmetric matrices — faster, more stable, returns real eigenvalues sorted in ascending order.
- **PCA** diagonalizes the covariance matrix: since it's symmetric, `V^(-1) = V^T` — computationally free.
- Symmetric positive definite matrices enable Cholesky decomposition.

### Python Exemplar

```python
import numpy as np

# ---- Build a symmetric matrix from R^T R ----
rng = np.random.default_rng(0)
R = rng.integers(1, 9, size=(4, 4)).astype(float)
A = R.T @ R    # always symmetric: (R^T R)^T = R^T R = A

print("A is symmetric:", np.allclose(A, A.T))

# ---- Use eigh (specialized for symmetric/Hermitian) ----
evals, V = np.linalg.eigh(A)   # returns REAL eigenvalues, sorted ascending
print("Eigenvalues (all real, non-negative):", np.round(evals, 4))
print("All non-negative:", np.all(evals >= -1e-10))

# ---- Verify orthogonality of eigenvectors ----
n = V.shape[1]
print("\nPairwise orthogonality check (should be ~0 for i≠j):")
for i in range(n):
    for j in range(i+1, n):
        dot = np.dot(V[:, i], V[:, j])
        print(f"  v{i}·v{j} = {dot:.2e}")

# V^T V = I (eigenvectors are orthonormal)
print("\nV^T V ≈ I:", np.allclose(V.T @ V, np.eye(n)))
print("V is orthogonal (V^(-1) == V^T):", np.allclose(np.linalg.inv(V), V.T))

# ---- Spectral theorem: A = V Λ V^T (NOT V Λ V^(-1)) ----
Lambda = np.diag(evals)
A_reconstructed = V @ Lambda @ V.T    # use V^T, not V^(-1)
print("\nSpectral theorem A = V Λ V^T :", np.allclose(A, A_reconstructed))

# ---- Non-symmetric matrix can have complex eigenvalues ----
B = np.array([[0, -1],
              [4,  0]], dtype=float)
print("\nB is symmetric:", np.allclose(B, B.T))
evals_B, _ = np.linalg.eig(B)
print("B's eigenvalues:", evals_B)   # complex: ±2i
print("All real?", np.all(np.isreal(evals_B)))   # False

# ---- Monte Carlo: 1000 symmetric matrices → always real eigenvalues ----
all_real_count = 0
for _ in range(1000):
    X = rng.standard_normal((5, 5))
    S = X.T @ X    # symmetric by construction
    evs, _ = np.linalg.eigh(S)
    if np.all(np.isreal(evs)):
        all_real_count += 1
print(f"\n{all_real_count}/1000 symmetric matrices have all-real eigenvalues")
```

---

## 10. Singular Matrices & Rank

### Concept

A matrix is **singular** if `det(A) = 0`. This means it has a **non-trivial null space** — there exists a non-zero vector `v` such that `Av = 0`.

**Connecting the dots:**
- `det(A) = Πᵢ λᵢ` → if `det = 0` → at least one `λᵢ = 0`
- `Av = 0 = 0·v` → the zero eigenvalue's eigenvector is in the null space
- Rank deficiency = linearly dependent columns = collapsed dimension

**Chain of equivalences for a singular matrix:**
```
Linear dependency → Rank deficiency → det(A) = 0 → λᵢ = 0 for some i → Non-trivial null space
```

**Rank-eigenvalue relationship (for diagonalizable/symmetric matrices):**
```
rank(A) = number of non-zero eigenvalues (counting multiplicity)
```

### Core Formulas

```
det(A) = 0          ↔  A is singular
Av = 0 for some v≠0  ↔  0 is an eigenvalue
rank(A) = r         ↔  m - r eigenvalues equal zero (for m×m matrix)
```

### Geometric Meaning

A singular transformation **collapses at least one dimension**. The null space vectors represent the directions that get mapped to zero — the "lost information." Just as a flat shadow can't tell you the 3D shape that cast it, a singular matrix loses information irreversibly.

### ML Application

- Singular design matrix in linear regression → no unique solution → use pseudoinverse.
- Rank-deficient covariance matrix → multicollinear features → PCA needed.
- Zero eigenvalue in PCA → redundant dimension, can be dropped.
- `rank(A^T A) = rank(A)` → A^T A has same null space behavior as A.

### Python Exemplar

```python
import numpy as np
from scipy.linalg import null_space

# ---- Singular matrix: columns are linearly dependent ----
A = np.array([[1, 4, 7],
              [2, 5, 8],
              [3, 6, 9]], dtype=float)
# Column 3 = 2*column2 - column1

print("Matrix A:\n", A)
print("Rank:", np.linalg.matrix_rank(A))      # 2 (not 3)
print("Determinant:", np.linalg.det(A))       # ~0.0

# ---- Eigendecomposition of singular matrix ----
evals, evecs = np.linalg.eig(A)
print("\nEigenvalues:", np.round(evals, 6))   # one eigenvalue ~0

# Find the zero eigenvalue
zero_idx = np.argmin(np.abs(evals))
print(f"\nZero eigenvalue index: {zero_idx}, λ = {evals[zero_idx]:.2e}")

# The corresponding eigenvector should be in the null space
v_null = evecs[:, zero_idx]
Av = A @ v_null
print("Av (should be ≈ 0):", np.round(Av, 6))

# ---- Verify via explicit null space computation ----
ns = null_space(A)
print("\nNull space basis:\n", np.round(ns, 4))
print("A @ null_space ≈ 0:", np.allclose(A @ ns, 0, atol=1e-10))

# ---- Linear dependency check ----
print("\nColumn 3 = 2*col2 - col1?")
print(np.allclose(A[:, 2], 2*A[:, 1] - A[:, 0]))   # True

# ---- Rank = number of non-zero eigenvalues ----
threshold = 1e-10
nonzero_evals = np.sum(np.abs(evals) > threshold)
print(f"\nNon-zero eigenvalues: {nonzero_evals}")
print(f"Matrix rank: {np.linalg.matrix_rank(A)}")   # should match

# ---- Full rank vs rank deficient ----
for mat_desc, mat in [
    ("Full rank 3x3", np.array([[1,0,0],[0,2,0],[0,0,3]], dtype=float)),
    ("Rank 2  3x3", A),
]:
    evs = np.linalg.eig(mat)[0]
    print(f"\n{mat_desc}: eigenvalues = {np.round(evs, 2)}, "
          f"rank={np.linalg.matrix_rank(mat)}, det={np.linalg.det(mat):.2f}")
```

---

## 11. Quadratic Forms

### Concept

A **quadratic form** maps a vector `w` and a symmetric matrix `A` to a scalar:
```
Q(w) = w^T A w
```

This is a generalization of a quadratic polynomial. The key insight is that the **sign of the quadratic form is entirely determined by the eigenvalues of A**.

**Derivation via diagonalization:**
If `A = VΛV^T` (symmetric), let `z = V^T w`:
```
w^T A w = w^T V Λ V^T w = z^T Λ z = Σᵢ λᵢ zᵢ²
```
Since `zᵢ² ≥ 0` always, the sign of `Q(w)` depends only on the signs of the `λᵢ`.

**Expanded 2D form:**
```
w^T A w = A₁₁x² + 2A₁₂xy + A₂₂y²
```
(for symmetric A, since A₁₂ = A₂₁)

### Core Formula

```
Q(w) = w^T A w = Σᵢ λᵢ zᵢ²    where z = V^T w
```

The sign of Q for all w is determined by the signs of all λᵢ.

### Geometric Meaning (as 3D surface)

Plotting `z = w^T A w` as a function of `(w₁, w₂)`:
- **All λᵢ > 0:** Upward-opening bowl (elliptic paraboloid) — always positive.
- **Mixed λᵢ signs:** Saddle shape (hyperbolic paraboloid) — crosses z=0.
- **All λᵢ < 0:** Downward-opening bowl — always negative.

### ML Application

- **Loss function landscape**: the Hessian (matrix of second derivatives) defines a quadratic form. Positive definite Hessian → convex loss → unique minimum.
- **Mahalanobis distance**: `(x-μ)^T Σ⁻¹ (x-μ)` — a quadratic form used in Gaussian models.
- **Regularization**: L2 penalty `λ||w||²` = `w^T (λI) w` — a quadratic form with scalar eigenvalues.
- **Rayleigh quotient**: `R(w) = w^T A w / w^T w` — used to find extreme eigenvalues.

### Python Exemplar

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quadratic_form_3d(A, title, ax):
    w1_vals = np.linspace(-2, 2, 60)
    w2_vals = np.linspace(-2, 2, 60)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    # Evaluate w^T A w at each (w1, w2)
    # For 2x2 symmetric A: = A11*x^2 + 2*A12*x*y + A22*y^2
    Z = A[0,0]*W1**2 + 2*A[0,1]*W1*W2 + A[1,1]*W2**2
    evals = np.linalg.eigvalsh(A)
    ax.plot_surface(W1, W2, Z, cmap='coolwarm', alpha=0.7)
    ax.plot_surface(W1, W2, np.zeros_like(Z), alpha=0.2, color='gray')
    ax.set_title(f"{title}\nλ = {np.round(evals, 2)}")

fig = plt.figure(figsize=(15, 4))

A_pd  = np.array([[2, 0.5], [0.5, 3]])       # positive definite (all λ > 0)
A_indef = np.array([[-1, 2], [2, 1]])         # indefinite (mixed λ)
A_nd  = np.array([[-3, -0.5], [-0.5, -2]])    # negative definite (all λ < 0)

for i, (A, title) in enumerate([(A_pd, "Positive Definite"),
                                  (A_indef, "Indefinite"),
                                  (A_nd, "Negative Definite")]):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    quadratic_form_3d(A, title, ax)

plt.tight_layout()
plt.savefig("quadratic_forms_3d.png", dpi=80)
print("Saved quadratic_forms_3d.png")

# ---- Empirical verification via random vectors ----
print("\n--- Empirical sign test on 10,000 random vectors ---")
rng = np.random.default_rng(42)
for A, desc in [(A_pd, "Positive Definite"), (A_indef, "Indefinite"), (A_nd, "Negative Definite")]:
    evals = np.linalg.eigvalsh(A)
    samples = [rng.standard_normal(2) for _ in range(10000)]
    vals = [w @ A @ w for w in samples]
    print(f"\n{desc} (λ={np.round(evals,2)})")
    print(f"  Min Q(w) = {min(vals):.4f},  Max Q(w) = {max(vals):.4f}")
    print(f"  All positive: {all(v > 0 for v in vals)}")
    print(f"  All negative: {all(v < 0 for v in vals)}")

# ---- Diagonalization identity: w^T A w = Σ λᵢ zᵢ² ----
A = A_pd
evals, V = np.linalg.eigh(A)
w = np.array([1.5, -0.8])
z = V.T @ w   # change to eigenvector basis
Q_direct = w @ A @ w
Q_via_lambda = np.sum(evals * z**2)
print(f"\nDirect w^T A w         = {Q_direct:.6f}")
print(f"Via Σ λᵢ zᵢ²           = {Q_via_lambda:.6f}")
print(f"Match: {np.isclose(Q_direct, Q_via_lambda)}")
```

---

## 12. Matrix Definiteness

### Concept

**Matrix definiteness** classifies a symmetric matrix by the sign of its quadratic form `w^T A w` for all non-zero vectors `w`. Since `w^T A w = Σ λᵢ zᵢ²`, the classification depends entirely on the **signs of the eigenvalues**.

| Category | Eigenvalue condition | Q(w) for all w≠0 |
|----------|---------------------|-------------------|
| Positive Definite (PD)   | all λᵢ > 0       | always > 0 |
| Positive Semi-Definite (PSD) | all λᵢ ≥ 0  | ≥ 0 (zero possible) |
| Indefinite               | mixed signs      | can be + or − |
| Negative Semi-Definite   | all λᵢ ≤ 0      | ≤ 0 |
| Negative Definite (ND)   | all λᵢ < 0      | always < 0 |

**Fundamental theorem:** `A^T A` is **always positive semi-definite** for any matrix A.

**Proof:**
```
x^T (A^T A) x = (Ax)^T (Ax) = ||Ax||² ≥ 0
```
It equals zero only when `Ax = 0`, i.e., when x is in the null space of A.
Therefore, all eigenvalues of `A^T A` are non-negative.

**Key implication:** `A^T A` is positive **definite** if A has full column rank.

### Core Formulas

```
A positive semi-definite  ↔  all λᵢ ≥ 0  ↔  w^T A w ≥ 0  ∀w
A^T A is always PSD       (proof: ||Ax||² ≥ 0)
eigenvalues(A^T A) = (singular values of A)²
rank(A^T A) = rank(A)
```

**Cholesky decomposition** (only for PD matrices):
```
A = L L^T    where L is lower triangular
```

### Geometric Meaning

- **PD:** quadratic form is a bowl curving upward — unique minimum at origin — convex.
- **PSD:** flat in some directions (zero eigenvalues = flat floor).
- **Indefinite:** saddle — can rise and fall — saddle points in optimization.
- **ND:** bowl curving downward — unique maximum.

### ML Application

- **Optimization:** Hessian of loss function must be PSD at minimum (necessary condition). Indefinite Hessian → saddle point, not minimum.
- **Covariance matrices:** always PSD by construction. Negative eigenvalue = computation error.
- **Cholesky decomposition:** generates correlated random samples in Monte Carlo simulation. Requires PD matrix.
- **Neural networks:** checking definiteness of the Hessian reveals whether gradient descent is at a true minimum.

### Python Exemplar

```python
import numpy as np

def classify_definiteness(A):
    evals = np.linalg.eigvalsh(A)
    if np.all(evals > 1e-10):
        return "Positive Definite"
    elif np.all(evals >= -1e-10):
        return "Positive Semi-Definite"
    elif np.all(evals < -1e-10):
        return "Negative Definite"
    elif np.all(evals <= 1e-10):
        return "Negative Semi-Definite"
    else:
        return "Indefinite"

# ---- Test different matrices ----
test_cases = {
    "PD  [[3,1],[1,2]]":    np.array([[3,1],[1,2]], dtype=float),
    "Indef [[-1,2],[2,1]]": np.array([[-1,2],[2,1]], dtype=float),
    "ND  [[-3,-1],[-1,-2]]":np.array([[-3,-1],[-1,-2]], dtype=float),
}
for desc, A in test_cases.items():
    evals = np.linalg.eigvalsh(A)
    print(f"{desc}: λ={np.round(evals,2)} → {classify_definiteness(A)}")

# ---- A^T A is always PSD — proof by example ----
print("\n--- A^T A is always PSD ---")
rng = np.random.default_rng(0)
test_shapes = [
    ("Tall 5x3",  rng.standard_normal((5, 3))),
    ("Wide 3x5",  rng.standard_normal((3, 5))),
    ("Square 4x4",rng.standard_normal((4, 4))),
    ("Rank def.", np.vstack([rng.standard_normal(3), rng.standard_normal(3),
                             rng.standard_normal(3)])),
]
for name, A in test_shapes:
    S = A.T @ A
    evals = np.linalg.eigvalsh(S)
    all_nonneg = np.all(evals >= -1e-10)
    rank_A = np.linalg.matrix_rank(A)
    n_nonzero_evals = np.sum(evals > 1e-10)
    print(f"\n{name}: shape {A.shape}, rank={rank_A}")
    print(f"  A^T A eigenvalues: {np.round(evals, 4)}")
    print(f"  All ≥ 0: {all_nonneg}, class: {classify_definiteness(S)}")
    print(f"  Non-zero eigenvalues: {n_nonzero_evals} = rank(A): {rank_A == n_nonzero_evals}")

# ---- Cholesky decomposition (PD matrices only) ----
A_pd = np.array([[4, 2, 0],
                 [2, 5, 1],
                 [0, 1, 3]], dtype=float)
print("\n--- Cholesky decomposition ---")
print("A is PD:", classify_definiteness(A_pd))
L = np.linalg.cholesky(A_pd)   # A = L L^T
print("L (lower triangular):\n", np.round(L, 4))
print("L @ L^T ≈ A:", np.allclose(L @ L.T, A_pd))

# ---- Hessian analysis in optimization ----
# f(x,y) = 3x² + xy + 2y² → Hessian = [[6,1],[1,4]]
Hessian = np.array([[6, 1],
                    [1, 4]], dtype=float)
print("\nHessian of f=3x²+xy+2y²:")
print("  Definiteness:", classify_definiteness(Hessian))  # PD → unique minimum
```

---

## 13. Generalized Eigendecomposition

### Concept

**Standard eigendecomposition** solves `Av = λv` — how does A scale v?

**Generalized eigendecomposition** solves:
```
A v = λ B v
```
Here, the eigenvector is scaled by **matrix B times scalar λ**, not just the scalar. This measures how A behaves **relative to** B.

Think of it as: in which direction does A dominate relative to B?

**Generalized Rayleigh quotient:**
```
λ = v^T A v / v^T B v
```
(Compare to standard: `λ = v^T A v / v^T v`)

**Reduction to standard form** (when B is PD with Cholesky `B = CC^T`):
```
Ã = C^(-1) A C^(-T)
→ Ã ũ = λ ũ   (standard eigenvalue problem)
→ v = C^(-T) ũ   (recover generalized eigenvectors)
```

**Eigenvector orthogonality:** Under the B-inner product (not standard dot product):
```
vᵢ^T B vⱼ = δᵢⱼ    (B-orthogonality)
```

### Core Formula

```
A v = λ B v          (generalized eigenvalue problem)
λᵢ = vᵢ^T A vᵢ / vᵢ^T B vᵢ   (Rayleigh quotient)
```

Special case: when `B = I` → reduces to standard `Av = λv`.

### ML Application

- **Fisher's Linear Discriminant Analysis (LDA):** Maximize between-class variance relative to within-class variance:
  ```
  S_B v = λ S_W v
  ```
  where `S_B` = between-class scatter, `S_W` = within-class scatter. This is a generalized eigenvalue problem.
- **Canonical Correlation Analysis (CCA):** finding directions of maximal correlation between two datasets.
- **Mechanical vibrations:** natural frequencies from `K v = λ M v`.

### Python Exemplar

```python
import numpy as np
from scipy.linalg import eigh

rng = np.random.default_rng(7)

# ---- Construct symmetric positive definite matrices ----
def make_spd(n, rng):
    X = rng.standard_normal((n, n))
    S = X.T @ X + 0.1 * np.eye(n)   # add 0.1*I for strict PD
    return S

A = make_spd(4, rng)
B = make_spd(4, rng)

# ---- Solve generalized eigenvalue problem: A v = λ B v ----
evals, V = eigh(A, B)   # scipy.linalg.eigh handles symmetric PD B
print("Generalized eigenvalues:", np.round(evals, 4))
print("V (columns = generalized eigenvectors) shape:", V.shape)

# ---- Verify: A vᵢ = λᵢ B vᵢ ----
print("\nVerification A*v = λ*B*v:")
for i in range(len(evals)):
    lhs = A @ V[:, i]
    rhs = evals[i] * B @ V[:, i]
    print(f"  i={i}: match = {np.allclose(lhs, rhs, atol=1e-8)}")

# ---- B-orthogonality: vᵢ^T B vⱼ = δᵢⱼ ----
print("\nB-orthogonality check (vᵢ^T B vⱼ):")
n = len(evals)
for i in range(n):
    for j in range(n):
        val = V[:, i] @ B @ V[:, j]
        expected = 1.0 if i == j else 0.0
        ok = np.isclose(val, expected, atol=1e-8)
        print(f"  ({i},{j}): {val:.4f} [expected {expected:.0f}] {'OK' if ok else 'FAIL'}")

# ---- When B = I → reduces to standard eigendecomposition ----
I = np.eye(A.shape[0])
evals_gen, V_gen = eigh(A, I)     # GED with B=I
evals_std, V_std = eigh(A)        # standard symmetric eigendecomposition
print("\nGED with B=I matches standard eigh:",
      np.allclose(np.sort(evals_gen), np.sort(evals_std)))

# ---- Fisher's LDA (generalized eigenvalue formulation) ----
# Generate two-class data
n_per_class = 50
mu1, mu2 = np.array([0., 0.]), np.array([3., 1.])
data1 = rng.multivariate_normal(mu1, np.array([[1, 0.5],[0.5, 2]]), n_per_class)
data2 = rng.multivariate_normal(mu2, np.array([[1, -0.3],[-0.3, 1]]), n_per_class)

# Between-class scatter S_B = (μ₁-μ₂)(μ₁-μ₂)^T
m1_hat = data1.mean(axis=0)
m2_hat = data2.mean(axis=0)
diff = (m1_hat - m2_hat).reshape(-1, 1)
S_B = diff @ diff.T

# Within-class scatter S_W = Σ₁ + Σ₂
S_W = (np.cov(data1.T) * (n_per_class-1) +
       np.cov(data2.T) * (n_per_class-1)) / (2*n_per_class - 2)

# Solve S_B w = λ S_W w  (LDA generalized eigenproblem)
evals_lda, evecs_lda = eigh(S_B, S_W)
w_lda = evecs_lda[:, -1]   # largest generalized eigenvalue → best discriminant direction
print("\nFisher LDA direction:", np.round(w_lda / np.linalg.norm(w_lda), 4))
```

---

## 14. Principal Component Analysis (PCA)

### Concept

**PCA** finds the directions of maximum variance in a dataset. It answers: "What are the most important directions of variation?"

**Algorithm:**
1. Mean-center the data: `X̃ = X - μ`
2. Compute covariance matrix: `C = (1/n-1) X̃^T X̃`
3. Eigendecompose C: `C = V Λ V^T` (symmetric → orthogonal diagonalization)
4. **Principal components** = columns of V (eigenvectors of C)
5. **Variance explained** by each PC = corresponding eigenvalue
6. Project data: `Z = X̃ V_k` (keep only k largest)

A **scree plot** (eigenvalues in descending order) reveals the data's structure:
- Single dominant eigenvalue → one main pattern.
- Multiple similar eigenvalues → multiple independent patterns.

### Core Formulas

```
C = (1/(n-1)) X̃^T X̃           (covariance matrix)
C = V Λ V^T                    (eigendecomposition)
variance_explained_ratio = λᵢ / Σ λⱼ
Z = X̃ V_k                     (projection onto k principal components)
X̃ ≈ Z V_k^T                   (reconstruction / compression)
```

### Geometric Meaning

PCA rotates the coordinate system so that axes align with the directions of maximum variance. The first axis captures the most variance, the second captures the most remaining variance (and is orthogonal to the first), and so on. Dropping low-variance axes = dimensionality reduction with minimum information loss.

### ML Application

- **Dimensionality reduction**: compress high-dimensional features while preserving variance.
- **Visualization**: project 100D data to 2D for plotting.
- **Noise reduction**: low-variance PCs often capture noise; dropping them denoises.
- **Preprocessing**: decorrelates features (useful for algorithms that assume independence).

### Python Exemplar

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---- Scenario A: 1 dominant factor ----
n_samples = 200
n_features = 15

f1 = rng.standard_normal(n_samples).reshape(-1, 1)  # single latent factor
W1 = rng.standard_normal(n_features).reshape(1, -1)  # loadings
X_A = f1 @ W1 + 0.2 * rng.standard_normal((n_samples, n_features))

# ---- Scenario B: 3 independent factors ----
f2 = rng.standard_normal((n_samples, 3))
W2 = np.zeros((3, n_features))
W2[0, 0:5]  = rng.standard_normal(5)   # factor 1 → features 1-5
W2[1, 5:10] = rng.standard_normal(5)   # factor 2 → features 6-10
W2[2, 10:]  = rng.standard_normal(5)   # factor 3 → features 11-15
X_B = f2 @ W2 + 0.2 * rng.standard_normal((n_samples, n_features))

def run_pca(X, name):
    # Step 1: mean center
    X_tilde = X - X.mean(axis=0)
    # Step 2: covariance matrix
    C = (X_tilde.T @ X_tilde) / (X.shape[0] - 1)
    # Step 3: eigendecompose (eigh for symmetric)
    evals, V = np.linalg.eigh(C)
    # Sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    V = V[:, idx]
    # Variance explained
    total_var = evals.sum()
    explained_ratio = evals / total_var
    explained_cumsum = np.cumsum(explained_ratio)

    print(f"\n=== {name} ===")
    print(f"Top 5 eigenvalues: {np.round(evals[:5], 2)}")
    print(f"Variance by PC1: {explained_ratio[0]*100:.1f}%")
    print(f"Variance by PC1+2+3: {explained_cumsum[2]*100:.1f}%")

    return evals, explained_ratio, V

evals_A, ratio_A, V_A = run_pca(X_A, "Scenario A (1 factor)")
evals_B, ratio_B, V_B = run_pca(X_B, "Scenario B (3 factors)")

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, evals, ratio, title in [
    (axes[0], evals_A, ratio_A, "Scenario A: 1 dominant factor"),
    (axes[1], evals_B, ratio_B, "Scenario B: 3 factors"),
]:
    ax.bar(range(1, len(evals)+1), ratio * 100)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title(title)
plt.tight_layout()
plt.savefig("pca_scree_plots.png", dpi=80)
print("\nSaved pca_scree_plots.png")

# ---- Full PCA pipeline: project & reconstruct ----
X = X_B.copy()
X_tilde = X - X.mean(axis=0)
C = (X_tilde.T @ X_tilde) / (X.shape[0] - 1)
evals, V = np.linalg.eigh(C)
idx = np.argsort(evals)[::-1]
V = V[:, idx]

k = 3   # keep 3 principal components
V_k = V[:, :k]

Z = X_tilde @ V_k            # project to k-dimensional space
X_reconstructed = Z @ V_k.T  # reconstruct (approx)

reconstruction_error = np.linalg.norm(X_tilde - X_reconstructed, 'fro')
print(f"\nReconstruction error with k={k} PCs: {reconstruction_error:.4f}")
print(f"Original Frobenius norm:              {np.linalg.norm(X_tilde, 'fro'):.4f}")
```

---

## 15. Covariance & Correlation Matrices

### Concept

A **covariance matrix** captures pairwise relationships between all features simultaneously:
```
C = (1/(n-1)) X̃^T X̃
```
where `X̃` is the mean-centered data matrix (n samples × p features).

**Element interpretation:**
- `Cᵢᵢ` = variance of feature i
- `Cᵢⱼ` = covariance between features i and j (how they move together)

**Key properties:**
- Always symmetric: `Cᵢⱼ = Cⱼᵢ`
- Always PSD: `x^T C x ≥ 0` for all x (because `C = X̃^T X̃ / (n-1)` and we proved `A^T A` is PSD)
- Diagonal elements (variances) are always non-negative

**Correlation matrix** = normalized covariance (scale-invariant):
```
ρᵢⱼ = Cᵢⱼ / √(Cᵢᵢ · Cⱼⱼ)
```
Diagonal is all 1s. Off-diagonal values in [-1, 1].

### Core Formulas

```
C = (1/(n-1)) X̃^T X̃         (covariance matrix, p×p)
Cᵢⱼ = (1/(n-1)) Σₖ (xₖᵢ - x̄ᵢ)(xₖⱼ - x̄ⱼ)
ρᵢⱼ = Cᵢⱼ / √(Cᵢᵢ Cⱼⱼ)     (correlation coefficient)
```

### ML Application

- **PCA** diagonalizes the covariance matrix.
- **Gaussian Naive Bayes**: models `p(x | class)` with per-class covariance matrices.
- **Multivariate Gaussian**: `f(x) = (2π)^(-p/2) |Σ|^(-1/2) exp(-½(x-μ)^T Σ^(-1)(x-μ))`
- **Mahalanobis distance**: `d = √((x-μ)^T Σ^(-1)(x-μ))` — accounts for feature correlations.

### Python Exemplar

```python
import numpy as np

rng = np.random.default_rng(99)

# ---- Simulate multivariate data ----
n, p = 100, 5
# Create correlated features: X = Z @ L^T where L controls correlations
true_cov = np.array([[1.0, 0.8, 0.0, 0.0, 0.0],
                     [0.8, 1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.6, 0.0],
                     [0.0, 0.0, 0.6, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 0.0, 1.0]])
L = np.linalg.cholesky(true_cov)
X = rng.standard_normal((n, p)) @ L.T   # correlated data

# ---- Compute covariance matrix (vectorized) ----
X_tilde = X - X.mean(axis=0)           # mean-center each feature
C_matrix = (X_tilde.T @ X_tilde) / (n - 1)   # p×p covariance
C_numpy  = np.cov(X.T)                 # NumPy reference
print("Custom C matches np.cov:", np.allclose(C_matrix, C_numpy))
print("\nCovariance matrix:\n", np.round(C_matrix, 3))

# Properties
print("\nSymmetric:", np.allclose(C_matrix, C_matrix.T))
evals_C = np.linalg.eigvalsh(C_matrix)
print("Eigenvalues (all ≥ 0):", np.round(evals_C, 4))
print("PSD:", np.all(evals_C >= -1e-10))

# ---- Correlation matrix (scale-invariant) ----
std_devs = np.sqrt(np.diag(C_matrix))   # σᵢ = √Cᵢᵢ
outer_stds = np.outer(std_devs, std_devs)
Corr = C_matrix / outer_stds
print("\nCorrelation matrix:\n", np.round(Corr, 3))
print("Diagonal of correlation matrix:", np.round(np.diag(Corr), 6))  # all 1s

# ---- Mahalanobis distance ----
C_inv = np.linalg.inv(C_matrix)
mu = X.mean(axis=0)
test_point = X[0]
mahal_dist = np.sqrt((test_point - mu) @ C_inv @ (test_point - mu))
euclidean_dist = np.linalg.norm(test_point - mu)
print(f"\nEuclidean distance of X[0] from mean: {euclidean_dist:.4f}")
print(f"Mahalanobis distance (accounts for correlation): {mahal_dist:.4f}")

# ---- Connection to A^T A (PSD proof) ----
# C = (1/n-1) X_tilde^T X_tilde
# This is (1/n-1) * (form A^T A with A = X_tilde)
# Therefore C is always PSD
print("\nC = (1/(n-1)) X_tilde^T X_tilde")
print("A^T A is always PSD → covariance matrix is always PSD. QED.")
```

---

## 16. Geometric Transformations

### Concept

Linear transformations (represented by matrices) include:
- **Rotation:** `R(θ)` — rotates by angle θ, preserves lengths and angles, `det = 1`
- **Scaling:** `diag(sx, sy)` — scales independently along axes
- **Shearing:** off-diagonal entries — parallelogram distortion
- **Reflection:** orthogonal matrix with `det = -1`
- **Projection:** projects onto a subspace, `P² = P`

**Composition:** applying T₁ then T₂ = matrix product `T₂ T₁` (right to left).

**Inverse transformation:** `T⁻¹` reverses the transformation. `T⁻¹T = I`.

**Rotation matrix (2D):**
```
R(θ) = [[cos θ, -sin θ],
         [sin θ,  cos θ]]
```
- Columns are orthogonal unit vectors (`R^T R = I`)
- Pythagorean identity: `cos²θ + sin²θ = 1` → unit norm columns
- `det(R) = 1` → pure rotation, no area change

### ML Application

- **Data augmentation:** rotate/flip images during training.
- **Feature normalization:** affine transformation `x̂ = (x - μ) / σ`.
- **Neural network layers:** linear layers = matrix multiplication = linear transformation.
- **Computer vision:** affine transforms for image alignment and perspective correction.

### Python Exemplar

```python
import numpy as np
import matplotlib.pyplot as plt

# ---- Rotation matrix and its properties ----
theta = np.pi / 5   # 36 degrees clockwise
R = np.array([[np.cos(theta),  np.sin(theta)],
              [-np.sin(theta), np.cos(theta)]])  # clockwise convention

print("R^T R (should be I):\n", np.round(R.T @ R, 10))
print("R R^T (should be I):\n", np.round(R @ R.T, 10))
print("det(R) =", np.linalg.det(R))             # 1.0
print("R^(-1) == R^T:", np.allclose(np.linalg.inv(R), R.T))

# ---- Points defining a house polygon ----
house = np.array([[0, 0, 1, 2, 2, 0],   # x-coords
                  [0, 2, 3, 2, 0, 0]])   # y-coords (2 x 6)

house_rotated = R @ house
house_recovered = R.T @ house_rotated    # apply inverse = transpose

print("\nOriginal ≈ Recovered:", np.allclose(house, house_recovered))

# ---- Distance preservation ----
distances_original = np.linalg.norm(house, axis=0)
distances_rotated  = np.linalg.norm(house_rotated, axis=0)
print("Distances preserved by rotation:",
      np.allclose(distances_original, distances_rotated))

# ---- Geometric transformation composition ----
# Scale by 2 in x, 3 in y; then rotate 45 degrees
S = np.diag([2.0, 3.0])             # scaling matrix
R45 = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                [np.sin(np.pi/4),  np.cos(np.pi/4)]])
T_composed = R45 @ S               # apply S first, then R45

x = np.array([1.0, 0.0])
print("\nOriginal x:", x)
print("After scale then rotate:", T_composed @ x)
print("After scale alone:      ", S @ x)
print("After scale then rotate (step by step):", R45 @ (S @ x))

# ---- Verify R45 @ S ≠ S @ R45 (matrix multiplication non-commutative) ----
print("R45 @ S == S @ R45?", np.allclose(R45 @ S, S @ R45))   # False

# ---- Visualize transformation ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, pts, title in [
    (axes[0], house, "Original"),
    (axes[1], house_rotated, "Rotated 36°"),
    (axes[2], house_recovered, "Recovered (R^T applied)")
]:
    ax.fill(pts[0], pts[1], alpha=0.3)
    ax.plot(np.append(pts[0], pts[0,0]), np.append(pts[1], pts[1,0]), 'b-o')
    ax.set_aspect('equal'); ax.set_title(title); ax.grid(True)
plt.tight_layout()
plt.savefig("geometric_transformations.png", dpi=80)
print("Saved geometric_transformations.png")
```

---

## 17. Image Filtering & Convolution

### Concept

**Image filtering** applies a kernel (filter matrix) K to an image I by sliding it across every pixel position and computing a **Hadamard product + sum** (element-wise multiply then add):

```
output(r, c) = ΣᵢΣⱼ K(i,j) · I(r+i, c+j)
```

This is a 2D **discrete convolution** — the same dot product operation as matrix multiplication, but applied to 2D patches.

**Gaussian smoothing kernel** (removes noise):
```
G(x, y) = exp(-(x² + y²) / σ²)
```
σ controls the smoothing width. Larger σ → more blurring.

**Edge detection** via gradient magnitude:
```
gradient_magnitude = √((∂I/∂x)² + (∂I/∂y)²)
```
Horizontal/vertical Sobel kernels compute these partial derivatives.

### ML Application

- **Convolutional Neural Networks (CNNs):** the convolution layer is exactly this sliding kernel operation.
- **Feature extraction:** kernels detect edges, textures, patterns.
- **Gaussian smoothing** is a preprocessing step to reduce noise before feature detection.

### Python Exemplar

```python
import numpy as np
from scipy.ndimage import convolve

# ---- Gaussian smoothing kernel ----
sigma = 1.5
grid = np.linspace(-3, 3, 21)
X, Y = np.meshgrid(grid, grid)
G = np.exp(-(X**2 + Y**2) / sigma**2)
G /= G.sum()   # normalize so kernel sums to 1 (preserves brightness)

print("Gaussian kernel (center 5x5 portion):\n",
      np.round(G[8:13, 8:13], 4))
print("Kernel sum (should be ~1):", G.sum())

# ---- Create a synthetic test image with noise ----
rng = np.random.default_rng(0)
image = np.zeros((50, 50))
image[15:35, 15:35] = 1.0      # white square
image += 0.2 * rng.standard_normal((50, 50))   # add Gaussian noise

# ---- Apply Gaussian smoothing ----
image_smoothed = convolve(image, G)

# ---- Edge detection via Sobel kernels ----
K_h = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], dtype=float)   # horizontal gradient (Sobel x)
K_v = K_h.T                                # vertical gradient (Sobel y)

edges_h = convolve(image_smoothed, K_h)
edges_v = convolve(image_smoothed, K_v)
# Gradient magnitude: sqrt(∂x² + ∂y²)
edge_magnitude = np.sqrt(edges_h**2 + edges_v**2)

print(f"\nOriginal image shape: {image.shape}")
print(f"Smoothed image shape: {image_smoothed.shape}")
print(f"Edge magnitude range: [{edge_magnitude.min():.3f}, {edge_magnitude.max():.3f}]")

# ---- Manual 2D convolution (no scipy) ----
def convolve2d_manual(image, kernel):
    ih, iw = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    # Zero-pad the image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)
    for r in range(ih):
        for c in range(iw):
            patch = padded[r:r+kh, c:c+kw]
            output[r, c] = np.sum(patch * kernel)   # Hadamard product + sum
    return output

small_img = image[:10, :10]
K_small = np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]], dtype=float)  # Laplacian kernel

out_manual = convolve2d_manual(small_img, K_small)
out_scipy  = convolve(small_img, K_small)
print("\nManual conv ≈ scipy conv:", np.allclose(out_manual, out_scipy))
```

---

## 18. Numerical Stability & Condition Number

### Concept

**Numerical stability** describes how sensitive a computation is to small perturbations in the input. The **condition number** quantifies this sensitivity for matrix operations:

```
κ(A) = ||A|| · ||A⁻¹|| = σ_max / σ_min
```

where `σ_max` and `σ_min` are the largest and smallest singular values.

**Interpretation:**
- `κ ≈ 1`: perfectly conditioned — small input errors → small output errors.
- `κ ≈ 10^k`: k digits of accuracy lost.
- `κ ≫ 1` ("ill-conditioned"): solving `Ax = b` gives unreliable results.

**Hilbert matrix** (classical ill-conditioned example):
```
H_ij = 1 / (i + j - 1)
```
The Hilbert matrix has an astronomically large condition number — inversion becomes completely numerically unreliable beyond size ~10.

**Practical implications:**
- Use `np.linalg.solve` instead of `inv(A) @ b`.
- For symmetric systems: use `eigh` instead of `eig`.
- Use QR or Cholesky decomposition instead of direct inversion.
- **Never compute A⁻¹ explicitly** unless you actually need the inverse matrix itself.

### Core Formulas

```
κ(A) = σ_max / σ_min            (condition number via SVD)
κ(A) = ||A|| · ||A⁻¹||          (general definition)
error_amplification ≤ κ(A) · (relative_perturbation_in_b / relative_norm_b)
```

### ML Application

- **Gram matrix X^T X**: condition number explodes with multicollinear features → Ridge regression adds `λI` to reduce conditioning: `κ(X^T X + λI)` much smaller.
- **Gradient descent convergence:** depends on condition number of Hessian — large `κ` → slow convergence.
- **Neural network training:** poorly scaled inputs → ill-conditioned weight updates → instability.

### Python Exemplar

```python
import numpy as np
from scipy.linalg import hilbert

# ---- Condition number growth with Hilbert matrix size ----
print("=== Hilbert Matrix Condition Numbers ===")
print(f"{'Size':>6} | {'Condition Number':>20} | {'Digits Lost':>12}")
print("-" * 45)

for n in [3, 5, 8, 10, 12]:
    H = hilbert(n)
    kappa = np.linalg.cond(H)
    digits_lost = int(np.log10(kappa)) if kappa > 1 else 0
    print(f"{n:>6} | {kappa:>20.2e} | {digits_lost:>12}")

# ---- Well-conditioned vs ill-conditioned: solving Ax = b ----
b = np.array([1.0, 2.0])
b_perturbed = b + 1e-7 * np.array([1.0, -1.0])  # tiny perturbation

# Well-conditioned system (identity)
A_good = np.eye(2) * 2.0
x1 = np.linalg.solve(A_good, b)
x1_perturbed = np.linalg.solve(A_good, b_perturbed)
print(f"\n--- Well-conditioned (κ = {np.linalg.cond(A_good):.1f}) ---")
print(f"Solution change: {np.linalg.norm(x1 - x1_perturbed):.2e}")

# Ill-conditioned system (Hilbert 2x2)
A_bad = hilbert(2)
x2 = np.linalg.solve(A_bad, b)
x2_perturbed = np.linalg.solve(A_bad, b_perturbed)
print(f"\n--- Ill-conditioned (κ = {np.linalg.cond(A_bad):.2e}) ---")
print(f"Solution change: {np.linalg.norm(x2 - x2_perturbed):.2e}")

# ---- Ridge regression reduces condition number ----
X = np.array([[1, 2], [1, 2.01], [1, 1.99]], dtype=float)  # near-multicollinear
XtX = X.T @ X
print(f"\n--- Multicollinear X: κ(X^T X) = {np.linalg.cond(XtX):.2e} ---")
for lam in [0, 0.01, 0.1, 1.0]:
    regularized = XtX + lam * np.eye(2)
    print(f"  λ={lam:.2f}: κ(X^T X + λI) = {np.linalg.cond(regularized):.2e}")

# ---- eigh vs eig for symmetric matrices: stability ----
S = hilbert(8)
S_sym = (S + S.T) / 2   # force exact symmetry

evals_eig,  _ = np.linalg.eig(S_sym)   # general (not exploiting symmetry)
evals_eigh, _ = np.linalg.eigh(S_sym)  # symmetric-specific (more stable)

print(f"\nSmallest eigenvalue (eig):  {min(evals_eig.real):.4e}")
print(f"Smallest eigenvalue (eigh): {min(evals_eigh):.4e}")
print("eigh returns real-only sorted values: more reliable for PSD matrices")

# ---- When NOT to compute A^(-1) directly ----
rng = np.random.default_rng(3)
A = rng.standard_normal((100, 100))
b = rng.standard_normal(100)

x_solve = np.linalg.solve(A, b)            # GOOD: stable
x_inv   = np.linalg.inv(A) @ b            # WORKS but wastes memory and less stable

residual_solve = np.linalg.norm(A @ x_solve - b)
residual_inv   = np.linalg.norm(A @ x_inv   - b)
print(f"\nResidual (solve): {residual_solve:.2e}")
print(f"Residual (inv):   {residual_inv:.2e}")
print("Use np.linalg.solve, not inv(A)@b!")
```

---

## Quick Reference: Key Formulas

| Concept | Formula | When to use |
|---------|---------|-------------|
| Dot product | `a·b = Σ aᵢbᵢ` | Similarity, projections |
| Matrix inverse (2×2) | `A⁻¹ = 1/(ad-bc) [[d,-b],[-c,a]]` | Small closed-form inverse |
| Left inverse | `A_L⁻¹ = (A^T A)⁻¹ A^T` | Least squares (tall matrix) |
| Pseudoinverse | `A⁺ = pinv(A)` | Any shape, any rank |
| Characteristic polynomial | `det(A - λI) = 0` | Find eigenvalues |
| Eigenvector | `(A - λᵢI)v = 0` | Find directions |
| Diagonalization | `A = V Λ V⁻¹` | Symmetric: `A = V Λ V^T` |
| Spectral theorem | `A = A^T → A = V Λ V^T` | Symmetric matrices |
| Trace identity | `tr(A) = Σᵢ λᵢ` | Always holds |
| Determinant identity | `det(A) = Πᵢ λᵢ` | Always holds |
| Quadratic form | `Q = w^T A w = Σᵢ λᵢ zᵢ²` | Definiteness, optimization |
| A^T A is PSD | `x^T(A^T A)x = ‖Ax‖² ≥ 0` | Covariance matrices, regression |
| QR decomposition | `A = QR`, `Q^(-1) = Q^T`, `R^(-1)` via back-sub | Stable least squares |
| Covariance matrix | `C = X̃^T X̃ / (n-1)` | PCA, Gaussian models |
| Condition number | `κ(A) = σ_max / σ_min` | Numerical stability |
| Generalized eigenvalue | `Av = λBv` | LDA, CCA |

---

## Roadmap: Learning Order

```
Vectors & Norms
    ↓
Matrix Operations (multiply, transpose, trace, rank)
    ↓
Determinants (2×2 formula, cofactor expansion)
    ↓
Matrix Inverse (conditions, 2×2 shortcut, left/right/pseudo)
    ↓
Orthogonality & Gram-Schmidt
    ↓
QR Decomposition (A = QR, stable least squares)
    ↓
Eigenvalues & Eigenvectors (characteristic polynomial, null space)
    ↓
Matrix Diagonalization (A = VΛV⁻¹, trace/det identities)
    ↓
Symmetric Matrices (spectral theorem, A = VΛV^T)
    ↓
Singular Matrices (zero eigenvalue, null space, rank)
    ↓
Quadratic Forms (w^T A w = Σ λᵢ zᵢ²)
    ↓
Matrix Definiteness (5 categories, A^T A is PSD, Cholesky)
    ↓
Generalized Eigendecomposition (Av = λBv, Fisher LDA)
    ↓
PCA (covariance eigendecomposition, principal components)
    ↓
Numerical Stability (condition number, Hilbert matrices)
```

---

*Based on the "Machine Learning Mathematics" series (Demystify with Ajay, Dr. Rajendra). All Python examples use NumPy/SciPy and are independently runnable.*
