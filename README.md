# Distributed Logistic Regression using ADMM Algorithm

## Overview

This project implements **Distributed Logistic Regression** using the **Alternating Direction Method of Multipliers (ADMM)** algorithm with **Message Passing Interface (MPI)** for parallel processing. The implementation is designed to handle large-scale datasets distributed across multiple files and processes them efficiently using high-performance computing resources.

## 🧮 Algorithm & Mathematical Foundation

### ADMM (Alternating Direction Method of Multipliers)

ADMM is a powerful optimization algorithm that decomposes complex optimization problems into smaller, more manageable subproblems. For distributed logistic regression, we solve:

```
minimize: Σᵢ fᵢ(xᵢ) 
subject to: xᵢ = z for all i
```

Where:
- `fᵢ(xᵢ)` is the logistic loss function for dataset partition i
- `z` is the global consensus variable (shared coefficients)
- Each `xᵢ` represents local model parameters

### Logistic Regression Components

The implementation includes key logistic regression functions:

1. **Sigmoid Function** with numerical stability:
   ```
   σ(z) = 1 / (1 + e^(-z))
   ```

2. **Log Loss Function**:
   ```
   L(β) = -mean[y*z - log(1 + e^z)]
   ```

3. **Gradient Computation**:
   ```
   ∇L(β) = -X^T(y - σ(Xβ)) / n
   ```

4. **Hessian Matrix** for Newton's method optimization

## 🏗️ Code Structure & Logic

### Core Components

#### 1. **Data Processing** (`main()`)
- **Distributed File Loading**: Each MPI process handles a subset of CSV data files
- **Feature Standardization**: Global mean and standard deviation computed across all processes
- **Data Validation**: Robust error handling for different file formats

#### 2. **ADMM Updates** (`local_admm_update()`)
- **Local Optimization**: Each process solves a regularized logistic regression subproblem
- **Newton's Method**: Uses Hessian-based optimization with line search
- **Numerical Stability**: Handles singular matrices with gradient descent fallback

#### 3. **Global Consensus** 
- **Collective Communication**: Uses MPI Allreduce for global parameter averaging
- **Convergence Monitoring**: Tracks primal and dual residuals
- **Dual Variable Updates**: Maintains consistency across distributed components

#### 4. **Feature Standardization** (`standardize_features()`)
- **Distributed Statistics**: Computes global mean and variance across all processes
- **Numerical Stability**: Prevents division by zero in standardization

### Key Algorithms Implemented

1. **Distributed ADMM Algorithm**
2. **Newton's Method with Line Search**
3. **Numerically Stable Sigmoid Function**
4. **Global Feature Standardization**
5. **MPI-based Parallel Processing**

## 🚀 Technical Features

- **Parallel Processing**: MPI-based distributed computing
- **Scalable Architecture**: Handles multiple data files across processes
- **Robust Optimization**: Newton's method with gradient descent fallback
- **Convergence Guarantees**: Monitors primal and dual residuals
- **Memory Efficient**: Processes data in chunks
- **HPC Ready**: Designed for SLURM job scheduling systems

## 📁 Project Files

- **`logistic_regression_admm.py`**: Main implementation file containing the distributed ADMM algorithm
- **`run_admm.slurm`**: SLURM job submission script for HPC clusters
- **`README.md`**: This documentation file

## ⚡ Usage

### Prerequisites

```bash
# Required Python packages
pip install numpy pandas mpi4py
```

### Running on HPC Cluster

```bash
# Submit job to SLURM scheduler
sbatch run_admm.slurm
```

### Running Locally

```bash
# Run with MPI (example with 4 processes)
mpirun -n 4 python logistic_regression_admm.py
```

## 🔧 Configuration Parameters

- **`rho`**: ADMM penalty parameter (default: 2.0)
- **`max_iters`**: Maximum ADMM iterations (default: 150)
- **`tolerance`**: Convergence threshold (default: 1e-3)
- **`max_iter` (local)**: Local optimization iterations (default: 5)

## Algorithm Performance

The implementation provides:
- **Convergence Monitoring**: Tracks primal and dual residuals
- **Distributed Processing**: Scales with number of MPI processes
- **Robust Optimization**: Handles ill-conditioned problems
- **Memory Efficiency**: Processes large datasets in distributed manner

## Output

The program outputs:
- **Final Coefficients**: Intercept and feature coefficients
- **Convergence Information**: Iteration count and residuals
- **Processing Statistics**: Total samples, files, and processes used
- **Performance Metrics**: Processing time and resource utilization

## Mathematical Properties

- **Convergence**: ADMM guarantees convergence to global optimum for convex problems
- **Scalability**: Linear scaling with number of processors
- **Robustness**: Handles numerical instabilities in logistic regression
- **Consistency**: Maintains statistical consistency across distributed data

## Academic Context

This implementation demonstrates advanced concepts in:
- **Distributed Optimization**
- **Parallel Computing**
- **Machine Learning**
- **Numerical Methods**
- **High-Performance Computing**

---

*This project showcases the integration of mathematical optimization, distributed computing, and machine learning for solving large-scale logistic regression problems efficiently.*
