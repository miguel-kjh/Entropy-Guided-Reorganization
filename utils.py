import torch
import numpy as np

def lightweight_entropy_estimator(tensor, device="cuda"):
    """
    Lightweight entropy estimator using variances of features.

    Parameters:
    - tensor: torch.Tensor of shape (n, d), where n is the number of samples and d is the feature dimension.
    - device: str, the device to perform computations ('cpu' or 'cuda').

    Returns:
    - entropy: float, estimated entropy.
    """
    tensor = tensor.to(device)
    # reshape tensor to (n, d)
    if tensor.dim() > 2:
        tensor = tensor.view(-1, tensor.size(-1))

    # Compute variances of each feature (column)
    variances = torch.var(tensor, dim=0, unbiased=False)

    # Estimate entropy using the variances
    entropy = 0.5 * torch.sum(torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0, device=device)) * variances))
    return entropy.item()

def parzen_entropy_estimator(tensor, bandwidth=1.0, device="cuda"):
    """
    Entropy estimator using Parzen window (Kernel Density Estimation).

    Parameters:
    - tensor: torch.Tensor of shape (n, d), where n is the number of samples and d is the feature dimension.
    - bandwidth: float, the bandwidth of the Gaussian kernel.
    - device: str, the device to perform computations ('cpu' or 'cuda').

    Returns:
    - entropy: float, estimated entropy.
    """
    tensor = tensor.to(device)
    tensor = tensor.view(-1, tensor.size(-1))
    n, d = tensor.size()

    # Pairwise distances with Gaussian kernel
    pairwise_distances = torch.cdist(tensor, tensor, p=2) ** 2
    kernel_matrix = torch.exp(-pairwise_distances / (2 * bandwidth ** 2))

    # Average over the kernel matrix
    kernel_density = kernel_matrix.mean(dim=1)

    # Estimate entropy
    entropy = -torch.mean(torch.log(kernel_density + 1e-10))  # Add epsilon to avoid log(0)
    return entropy.item()

def pca_entropy_estimator(tensor, beta=1.0, device="cuda"):
    """
    Entropy estimator using PCA (singular value decomposition).

    Parameters:
    - tensor: torch.Tensor of shape (n, d), where n is the number of samples and d is the feature dimension.
    - beta: float, scaling parameter for stability.
    - device: str, the device to perform computations ('cpu' or 'cuda').

    Returns:
    - entropy: float, estimated entropy.
    """
    tensor = tensor.to(device)
    tensor = tensor.view(-1, tensor.size(-1))
    n, d = tensor.size()

    # Compute covariance matrix
    covariance_matrix = (1 / n) * torch.matmul(tensor.T, tensor)

    # Compute eigenvalues using SVD
    _, singular_values, _ = torch.svd(covariance_matrix)

    # Stabilize singular values
    singular_values = singular_values + beta

    # Estimate entropy
    entropy = 0.5 * torch.sum(torch.log(singular_values))
    return entropy.item()



if __name__ == "__main__":

    def gaussian_entropy_theoretical(cov_matrix):
        """
        Compute the theoretical entropy of a Gaussian distribution given its covariance matrix.
        
        Parameters:
        - cov_matrix: np.ndarray of shape (d, d), covariance matrix.
        
        Returns:
        - entropy: float, theoretical entropy of the Gaussian.
        """
        d = cov_matrix.shape[0]  # Dimensionality
        determinant = np.linalg.det(cov_matrix + 1e-10 * np.eye(d))  # Add small noise for stability
        entropy = 0.5 * np.log((2 * np.pi * np.e) ** d * determinant)
        return entropy

    # Generate synthetic data with a Gaussian distribution
    n_samples, n_features = 100, 50
    mean = np.zeros(n_features)
    cov_matrix = np.diag(np.random.rand(n_features))  # Diagonal covariance matrix
    data = np.random.multivariate_normal(mean, cov_matrix, size=n_samples)
    tensor_data = torch.tensor(data, dtype=torch.float32)

    # Theoretical entropy
    theoretical_entropy = gaussian_entropy_theoretical(cov_matrix)

    # Lightweight entropy estimator
    print("Lightweight entropy estimator")
    estimated_entropy = lightweight_entropy_estimator(tensor_data)

    print(theoretical_entropy, estimated_entropy)

    # Parzen entropy estimator
    print("Parzen entropy estimator")
    bandwidth = 1.0
    estimated_entropy = parzen_entropy_estimator(tensor_data, bandwidth=bandwidth)

    print(theoretical_entropy, estimated_entropy)

    # PCA entropy estimator
    print("PCA entropy estimator")
    beta = 1.0
    estimated_entropy = pca_entropy_estimator(tensor_data, beta=beta)

    print(theoretical_entropy, estimated_entropy)
