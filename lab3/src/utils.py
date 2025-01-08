import matplotlib.pyplot as plt
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data(file_path):
    """
    加载数据，返回特征矩阵、标签和特征数量
    :param file_path:
    :return: 特征矩阵、标签和特征数量
    """
    labels = []
    features = []
    max_index = 0  # 用于推断特征数量

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            labels.append(float(parts[0]))
            feature = []
            for item in parts[1:]:
                index, value = map(int, item.split(':'))
                feature.append((index - 1, value))
                max_index = max(max_index, index)
            features.append(feature)

    num_features = max_index

    # 将特征转换为稠密矩阵
    feature_matrix = torch.zeros(len(features), num_features, device=device)
    for i, feature in enumerate(features):
        for index, value in feature:
            feature_matrix[i, index] = value  # 根据索引赋值

    return feature_matrix, torch.tensor(labels).to(device), num_features


class LogisticRegression:
    """
    逻辑回归模型
    """
    def __init__(self, num_features, lambd):
        self.weights = torch.zeros(num_features, device=device, requires_grad=True)
        self.lambd = lambd

    def loss(self, X, y):
        """计算目标函数值"""
        logits = X @ self.weights
        logistic_loss = torch.mean(torch.log(1 + torch.exp(-y * logits)))
        regularization = self.lambd * torch.norm(self.weights) ** 2
        return logistic_loss + regularization

    def gradient(self, X, y):
        """计算梯度"""
        m = X.shape[0]
        logits = X @ self.weights
        probabilities = 1 / (1 + torch.exp(-y * logits))
        grad = -(1 / m) * torch.matmul(X.T, (y * (1 - probabilities))) + 2 * self.lambd * self.weights
        return grad

    def hessian(self, X, y):
        """计算海森矩阵"""
        m = X.shape[0]
        logits = X @ self.weights
        probabilities = 1 / (1 + torch.exp(-y * logits))
        diag_hessian = probabilities * (1 - probabilities)  # 对角线元素
        H = (1 / m) * torch.matmul(X.T, (X * diag_hessian.view(-1, 1))) + 2 * self.lambd * torch.eye(X.shape[1],                                                                                        device=device)
        return H


def backtracking_line_search(model, X, y, alpha=1.0, gamma=0.5, c=1e-4, alpha_min=1e-8):
    """
    执行带回溯线搜索的梯度下降，支持用户手动调整参数。
    model: LogisticRegression 模型
    X: 输入特征 (torch.Tensor)
    y: 标签 (torch.Tensor)
    alpha: 初始步长
    beta: 衰减因子，控制步长减少的速度
    c: 充分下降条件参数
    alpha_min: 步长搜索的下界，防止步长无限趋近于零
    model.weights: 优化后的权重
    history: 收敛历史数据，包括函数值、梯度范数和步长
    """
    max_iterations = 1000
    tol = 1e-3  # 梯度收敛阈值
    step_size = alpha  # 初始步长

    history = {"loss": [], "grad_norm": []}
    optimal_value = None
    ite = 0

    for iteration in range(max_iterations):
        ite = iteration
        loss = model.loss(X, y).item()
        optimal_value = loss
        grad = torch.autograd.grad(model.loss(X, y), model.weights, retain_graph=True)[0]
        grad_norm = torch.norm(grad).item()

        # 记录当前的损失、梯度范数和步长
        history["loss"].append(loss)
        history["grad_norm"].append(grad_norm)

        if grad_norm < tol:
            print(f"Converged at iteration {iteration}, Loss: {loss}")
            break

        step_size = alpha
        while step_size > alpha_min:
            new_weights = model.weights - step_size * grad
            with torch.no_grad():
                model.weights.copy_(new_weights)  # 临时更新权重计算新损失
            new_loss = model.loss(X, y)
            if new_loss <= loss - c * step_size * grad_norm:
                break
            step_size *= gamma

        if step_size < alpha_min:
            print(f"Line search failed at iteration {iteration}")
            break

        # 更新参数
        with torch.no_grad():
            model.weights -= step_size * grad

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss}")

    return ite, model.weights, history, optimal_value


def newton_optimization(model, X, y):
    """使用牛顿法优化模型"""
    max_iter = 1000
    tol = 1e-5
    history = {"loss": [], "grad_norm": []}  # 记录函数值和梯度范数
    optimal_value = None  # 用于存储最优目标函数值

    for iteration in range(max_iter):
        grad = model.gradient(X, y)
        grad_norm = torch.norm(grad).item()

        loss = model.loss(X, y)
        optimal_value = loss.item()
        print(f"Iteration {iteration}, Loss: {loss.item()}")
        history["loss"].append(loss.item())
        history["grad_norm"].append(grad_norm)

        if grad_norm < tol:
            print(f"Converged at iteration {iteration}, Loss: {loss.item()}")
            break

        H = model.hessian(X, y)
        p_k = torch.linalg.solve(H, -grad)
        with torch.no_grad():
            model.weights += p_k

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}")

    return model.weights, history, optimal_value


def plot_convergence(history, file_name1, file_name2):
    """
    绘制函数值、梯度范数和步长的收敛速度
    :param history:
    :return:
    """
    plt.figure(figsize=(8, 6))
    # plt.subplot(1, 2, 1)
    plt.semilogy(history["loss"], label="Loss - Optimal Value")
    plt.xlabel("Iterations")
    plt.ylabel("Log Scale: Loss Difference")
    plt.title("Convergence of Loss")
    plt.legend()
    plt.savefig(file_name1)

    # 梯度大小的收敛速度
    plt.figure(figsize=(8, 6))
    # plt.subplot(1, 2, 2)
    plt.semilogy(history["grad_norm"], label="Gradient Norm")
    plt.xlabel("Iterations")
    plt.ylabel("Log Scale: Gradient Norm")
    plt.title("Convergence of Gradient Norm")
    plt.legend()
    plt.savefig(file_name2)


def plot_step_size(history):
    """绘制线搜索参数和步长的关系"""
    plt.figure(figsize=(6, 4))
    plt.plot(history["step_size"], label="Step Size")
    plt.xlabel("Iterations")
    plt.ylabel("Step Size")
    plt.title("Step Size over Iterations")
    plt.legend()
    plt.show()


def test_parameters_and_plot(X, y, beta_values, c_values, lambd):
    """测试不同beta和c参数，并绘制与收敛速度的关系"""
    results = []

    # 遍历 beta 和 c 的组合
    for gamma in beta_values:
        for c in c_values:
            print(f"Testing beta={gamma}, c={c}")
            model = LogisticRegression(X.shape[1], lambd)  # 实例化模型
            # model.weights = torch.zeros_like(model.weights)  # 重置参数
            iterations, _, _ = backtracking_line_search(model, X, y, gamma=gamma, c=c)
            results.append((gamma, c, iterations))

    # 转换为 numpy 数组，方便绘图
    results = np.array(results)

    # 绘制 beta 和迭代次数的关系
    plt.figure(figsize=(10, 5))
    for c in c_values:
        subset = results[results[:, 1] == c]  # 筛选固定c的结果
        plt.plot(subset[:, 0], subset[:, 2], label=f"c={c}")
    plt.xlabel("Beta")
    plt.ylabel("Iterations to Converge")
    plt.title("Convergence Speed vs Beta for Different c Values")
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制 c 和迭代次数的关系
    plt.figure(figsize=(10, 5))
    for beta in beta_values:
        subset = results[results[:, 0] == beta]  # 筛选固定beta的结果
        plt.plot(subset[:, 1], subset[:, 2], label=f"Beta={beta}")
    plt.xlabel("c")
    plt.ylabel("Iterations to Converge")
    plt.title("Convergence Speed vs c for Different Beta Values")
    plt.legend()
    plt.grid()
    plt.show()
