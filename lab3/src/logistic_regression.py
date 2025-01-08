import numpy as np
import matplotlib.pyplot as plt

def load_sparse_data(file_path, num_features):
    """
    读取稀疏数据并转换为稠密矩阵。

    参数：
    file_path : str
        数据文件的路径（例如 a9a.txt）。
    num_features : int
        特征的总数（维度空间大小）。

    返回：
    X : ndarray
        稠密特征矩阵，形状为 (m, num_features)，m 是样本数量。
    y : ndarray
        标签向量，形状为 (m,)。
    """
    labels = []
    features = []

    # 逐行读取文件
    with open(file_path, 'r') as file:
        for line in file:
            # 分割每一行数据
            parts = line.strip().split()
            # 第一列是标签
            labels.append(int(parts[0]))
            # 剩余部分是特征:值对
            feature_row = np.zeros(num_features)
            for item in parts[1:]:
                index, value = item.split(':')
                feature_row[int(index) - 1] = float(value)  # 将索引转换为 0-based
            features.append(feature_row)
    
    # 转换为 numpy 数组
    X = np.array(features)
    y = np.array(labels)
    
    return X, y


# Logistic Regression 目标函数
def logistic_loss(X, y, x, lambda_):
    """
    计算逻辑回归的损失函数和梯度。

    参数：
    X : ndarray
        特征矩阵，形状为 (m, n)。
    y : ndarray
        标签向量，形状为 (m,)。
    x : ndarray
        参数向量，形状为 (n,)。
    lambda_ : float
        正则化参数。

    返回：
    loss : float
        逻辑回归的损失值。
    grad : ndarray
        损失的梯度，形状为 (n,)。
    """
    m = X.shape[0]
    Ax = np.dot(X, x)  # 计算 A*x
    z = y * Ax         # 元素逐个相乘
    c = np.exp(-z)     # 计算 exp(-b_i * a_i^T * x)
    
    # 损失函数
    loss = (1 / m) * np.sum(np.log(1 + c)) + lambda_ * np.dot(x, x)
    
    # 梯度
    b_div = y / (1 + c)  # b_i / (1 + exp(b_i * a_i^T * x))
    grad = - (1 / m) * np.dot(X.T, b_div) + 2 * lambda_ * x
    
    return loss, grad

# 回溯线搜索
def backtracking_line_search(X, y, x, d, grad, lambda_, alpha_0=1, gamma=0.5, c=1e-4):
    """
    执行回溯线搜索以找到步长。

    参数：
    X : ndarray
        特征矩阵，形状为 (m, n)。
    y : ndarray
        标签向量，形状为 (m,)。
    x : ndarray
        当前参数向量，形状为 (n,)。
    d : ndarray
        下降方向向量，形状为 (n,)。
    grad : ndarray
        梯度向量，形状为 (n,)。
    lambda_ : float
        正则化参数。
    alpha_0 : float
        初始步长。
    gamma : float
        步长衰减因子 (0 < gamma < 1)。
    c : float
        Armijo 条件参数 (0 < c < 1)。

    返回：
    alpha : float
        满足 Armijo 条件的步长。
    """
    alpha = alpha_0
    current_loss, _ = logistic_loss(X, y, x, lambda_)
    
    while True:
        x_new = x + alpha * d
        new_loss, _ = logistic_loss(X, y, x_new, lambda_)
        
        # Armijo 条件
        if new_loss <= current_loss + c * alpha * np.dot(grad, d):
            break
        
        alpha *= gamma  # 减小步长
    
    return alpha

# BFGS 优化
def logistic_regression_bfgs(X, y, lambda_, tol=1e-6, max_iter=1000):
    """
    使用 BFGS 优化逻辑回归。

    参数：
    X : ndarray
        特征矩阵，形状为 (m, n)。
    y : ndarray
        标签向量，形状为 (m,)。
    lambda_ : float
        正则化参数。
    tol : float
        用于判断收敛的梯度范数阈值。
    max_iter : int
        最大迭代次数。

    返回：
    x : ndarray
        最优参数向量，形状为 (n,)。
    losses : list
        每次迭代的损失值列表。
    grad_norms : list
        每次迭代的梯度范数列表。
    """
    m, n = X.shape
    x = np.zeros(n)  # 初始化参数
    H = np.eye(n)    # 初始化 Hessian 矩阵近似值
    
    losses = []
    grad_norms = []
    
    for iteration in range(max_iter):
        # 计算损失和梯度
        loss, grad = logistic_loss(X, y, x, lambda_)
        grad_norm = np.linalg.norm(grad)
        
        # 保存损失和梯度范数
        losses.append(loss)
        grad_norms.append(grad_norm)
        
        # 检查是否收敛
        if grad_norm < tol:
            print(f"Converged at iteration {iteration}")
            break
        
        # 计算下降方向
        d = -np.dot(H, grad)
        
        # 线搜索找到步长
        alpha = backtracking_line_search(X, y, x, d, grad, lambda_)
        
        # 更新参数
        x_new = x + alpha * d
        
        # 计算新的梯度
        _, grad_new = logistic_loss(X, y, x_new, lambda_)
        
        # BFGS 更新 Hessian 矩阵近似值
        s = x_new - x
        y_vec = grad_new - grad
        rho_denom = np.dot(y_vec, s)
        if np.abs(rho_denom) < 1e-10:  # 避免分母为 0
            print(f"Warning: Division by zero in BFGS update at iteration {iteration}. Skipping update.")
            x = x_new
            continue
        rho = 1.0 / rho_denom
        
        if rho > 0:
            H = (np.eye(n) - rho * np.outer(s, y_vec)) @ H @ (np.eye(n) - rho * np.outer(y_vec, s)) + rho * np.outer(s, s)
        
        # 更新变量
        x = x_new
    
    return x, losses, grad_norms


if __name__ == "__main__":
    # 数据文件路径和特征数量
    file_path = "../other/a9a.txt"  # 数据集路径
    num_features = 123  # 数据集的特征数量

    # 读取数据
    X, y = load_sparse_data(file_path, num_features)
    
    # 将标签转换为 {-1, +1}
    y = 2 * (y > 0) - 1

    # 特征标准化（零均值和单位方差）
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # 正则化参数
    lambda_ = 1e-4

    # 运行 BFGS 优化
    x_opt, losses, grad_norms = logistic_regression_bfgs(X, y, lambda_)

    # 绘制收敛曲线
    plt.figure()
    plt.semilogy(np.array(losses) - min(losses), label="目标函数值差")
    plt.xlabel("迭代次数")
    plt.ylabel("目标函数值差")
    plt.title("目标函数收敛曲线")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.semilogy(grad_norms, label="梯度范数")
    plt.xlabel("迭代次数")
    plt.ylabel("梯度范数")
    plt.title("梯度范数收敛曲线")
    plt.grid()
    plt.legend()
    plt.show()