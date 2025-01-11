import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

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





class LogisticRegression:
    """
    逻辑回归模型，支持 Backtracking Line Search 优化
    """
    def __init__(self, num_features, lambd=1e-4):
        """
        初始化逻辑回归模型
        参数:
        num_features : int
            特征数量
        lambd : float
            正则化参数
        """
        self.weights = np.zeros(num_features)  # 初始化参数为 0
        self.lambd = lambd  # 正则化参数

    def loss(self, X, y, weights=None):
        """
        计算逻辑回归的目标函数值
        参数:
        X : ndarray
            特征矩阵，形状为 (m, n)
        y : ndarray
            标签向量，形状为 (m,)
        weights : ndarray or None
            权重向量，形状为 (n,)。如果为 None，则使用 self.weights。
        返回:
        loss : float
            目标函数值
        """
        if weights is None:
            weights = self.weights

        m = X.shape[0]
        logits = np.dot(X, weights)
        logistic_loss = np.mean(np.log(1 + np.exp(-y * logits)))
        regularization = self.lambd * np.dot(weights, weights) / 2
        return logistic_loss + regularization

    def gradient(self, X, y):
        """
        计算梯度
        参数:
        X : ndarray
            特征矩阵，形状为 (m, n)
        y : ndarray
            标签向量，形状为 (m,)
        返回:
        grad : ndarray
            梯度向量，形状为 (n,)
        """
        m = X.shape[0]
        logits = np.dot(X, self.weights)
        probabilities = 1 / (1 + np.exp(-y * logits))
        grad = -(1 / m) * np.dot(X.T, y * (1 - probabilities)) + self.lambd * self.weights
        return grad

    def backtracking_line_search(self, X, y, grad, step_init=1.0, alpha=0.25, beta=0.8):
        """
        回溯线搜索来确定步长。
        """
        step = step_init
        current_loss = self.loss(X, y)
        direction = -grad  # 下降方向

        while True:
            new_weights = self.weights + step * direction
            new_loss = self.loss(X, y, weights=new_weights)

            if new_loss <= current_loss + alpha * step * np.dot(grad, direction):
                break

            step *= beta

            if step < 1e-8:
                print("Warning: Step size too small during line search.")
                break

        return step

    def optimize(self, X, y, step_init=1.0, max_iter=1000, tol=1e-4):
        """
        执行回溯线搜索的梯度下降优化
        """
        loss_history = []
        with tqdm(total=max_iter, desc="Optimizing", unit="iter") as pbar:
            for i in range(max_iter):
                loss = self.loss(X, y)
                grad = self.gradient(X, y)
                grad_norm = np.linalg.norm(grad)

                loss_history.append(loss)

                # 更新 tqdm 显示
                pbar.set_postfix({"Loss": f"{loss:.6f}", "Grad Norm": f"{grad_norm:.6f}"})
                pbar.update(1)

                # 检查收敛
                if grad_norm < tol:
                    print(f"Converged at iteration {i}.")
                    break

                # 计算步长并更新权重
                step = self.backtracking_line_search(X, y, grad, step_init=step_init)
                self.weights -= step * grad

        return loss_history

def plot_results(step_values, lambda_values, X, y):
    """
    绘制不同步长和正则化参数下的损失曲线
    """
    plt.figure(figsize=(16, 10))
    for k, step_init in enumerate(step_values):
        plt.subplot(2, len(step_values) // 2, k + 1)
        plt.title(f"Step Init: {step_init}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        for lambd in lambda_values:
            model = LogisticRegression(num_features=X.shape[1], lambd=lambd)
            loss_history = model.optimize(X, y, step_init=step_init)
            plt.plot(loss_history, label=f"λ={lambd}")

        plt.legend()
    plt.tight_layout()
    plt.show()


# 示例用法
if __name__ == "__main__":
    file_path = "../other/a9a.txt"
    num_features = 123
    X, y = load_sparse_data(file_path, num_features)

    y = 2 * (y > 0) - 1
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    step_values = [0.1, 0.3, 1.0, 3.0]
    lambda_values = [0, 1e-4, 1e-2, 1e-1]

    plot_results(step_values, lambda_values, X, y)