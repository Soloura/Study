# Causality

### [DirectLiNGAM](https://lingam.readthedocs.io/en/latest/tutorial/lingam.html)

DirectLiNGAM is a direct method for learning the basic LiNGAM model. It uses entropy-based measure to evaluate independence between error variables. The basic LiNGAM model makes the following assumptions.

1. Linearity
2. Non-Gaussian continuous error variables (except at most one)
3. Acyclicity
4. No hidden common causes

Denote observed variables by xi and error variables by ei and coefficients or connection strengths bij. Collect them in vectors x and e and a matrix B, respectivelly. Due to the acyclicity assumption, the adjacency matrix B can be permuted to be stricitly lower-triangular by a simultaneous row and column permutation. The error variabels ei are independent due to the assumption of no hidden common causes.

Then, mathematically, the model for observed variable vector x is written as `x = B * x + e`.

### [VARLiNGAM](https://lingam.readthedocs.io/en/latest/tutorial/var.html)

VARLiNGAM is an extension of the basic LiNGAM model to time series cases. It combines the basic LiNGAM model with the classic vector autoregressive models (VAR). It enables anaylzing both lagged and contemporaneous (instantaneous) causal relations, whereas the classic VAR only analyzes lagged causal relations. This VARLiNGAM makes the following assumptions similarly to the basic LiNGAM model:

1. Linearity
2. Non-Gaussian continuous error variables (except at most one)
3. Acyclicity of contemporaneous causal relations
4. No hidden common causes

Denote observed variables at time point t by xi(t) and error variables by ei(t). Collect them in vectors x(t) and e(t), respectivelly. Further, denote by matrices Bk adjacency matrices with time lag k.

Due to the acyclicity assumption of contemporanceous causal relations, the coefficient matrix B0 can be permuted to be strictly lower-triangular by a simultaneous row and column permutation. The error variables ei(t) are independent due to the assumption of no hidden common causes.

Then, mathematically, the model for observed variable vector x(t) is written as

`x(t) = sum( Bk * x(t-k) ) + e(t)`.

---

### Reference
- DirectLiNGAM, https://lingam.readthedocs.io/en/latest/tutorial/lingam.html, 2024-10-14-Mon.
- VARLiNGAM, https://lingam.readthedocs.io/en/latest/tutorial/var.html, 2024-10-14-Mon.
