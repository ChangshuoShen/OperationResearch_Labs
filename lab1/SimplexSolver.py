import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False


class DataGenerator:
    """
    数据生成类，用于生成随机线性规划问题
    """
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def generate_random_input(self, m=None):
        """
        随机生成矩阵 A、向量 b 和 c
        :param m: 可选输入变量数 m，若为 None 则随机生成
        :return: A, b, c
        """
        if m is None:
            m = random.randint(10, 200)  # 随机生成 m，范围在 [10, 200]
        n = m - random.randint(0, m // 3)  # 保证 n <= m 且 m - n <= m / 3

        # 生成矩阵和向量
        A = np.random.uniform(-10, 10, (n, m))
        x = np.random.uniform(0, 10, m)  # 随机生成解 x
        b = A @ x  # 保证 b = A * x 可行
        c = np.random.uniform(-10, 10, m)  # 随机目标函数系数

        return A, b, c

    def save_lp_problem(self, A, b, c, input_file, output_file):
        """
        将生成的线性规划问题保存到文件
        :param A: 系数矩阵
        :param b: 右端向量
        :param c: 目标函数系数
        :param input_file: 输入文件名
        :param output_file: 期望输出文件名
        """
        n, m = A.shape
        dot_product = np.dot(c, np.linalg.solve(A.T @ A, A.T @ b))  # 仅作为期望值参考
        result = [f"{n} {m}", " ".join(f"{value:.10f}" for value in c)]
        augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
        for row in augmented_matrix:
            result.append(" ".join(f"{value:.10f}" for value in row))

        with open(input_file, "w") as f:
            f.write("\n".join(result) + "\n")

        with open(output_file, "w") as f:
            f.write(f"{dot_product:.10f}\n")
            f.write(" ".join(f"{value:.10f}" for value in np.linalg.solve(A.T @ A, A.T @ b)) + "\n")


class SimplexSolver:
    """
    单纯形法求解器类
    """
    def __init__(self):
        self.No_feasible_solution = 0  # 无可行解计数
        self.No_unbounded_solution = 0  # 无界解计数

    @staticmethod
    def convert_to_standard_form(c, A, b):
        """
        转换为标准形式，移除全零行
        """
        zero_row_indices = [i for i in range(len(b)) if np.all(A[i, :] == 0)]
        A = np.delete(A, zero_row_indices, axis=0)
        b = np.delete(b, zero_row_indices, axis=0)
        return c, A, b

    @staticmethod
    def remove_redundant_constraints(A, b):
        """
        检查并移除冗余约束，使用 QR 分解
        """
        A = np.column_stack((A, b))
        q, r = np.linalg.qr(A.T)
        valid_indices = np.abs(np.diag(r)) > 1e-6
        A_reduced = (A.T[:, valid_indices]).T
        return A_reduced[:, :-1], A_reduced[:, -1]

    @staticmethod
    def initialize_feasible_solution(A, b, c):
        """
        使用大 M 法初始化一个可行基解
        """
        m, n = A.shape
        M = max((np.abs(c)).max() * 1e5, 1e6)
        artificial_vars = np.eye(m)
        A_aug = np.hstack([A, artificial_vars])
        c_aug = np.hstack([c, M * np.ones(m)])
        initial_basic = np.concatenate([np.zeros(n), b])  # 初始解
        return A_aug, c_aug, initial_basic, M

    def simplex_method(self, A, b, c, initial_basic, M):
        """
        单纯形法的迭代过程
        """
        if len(A.shape) == 1:
            A = A.reshape(1, -1)
        m, n = A.shape
        x = initial_basic
        var_index = np.arange((n - m), n)
        table = np.column_stack((A, b))
        zs = np.zeros(n + 1)
        zs[:-1] = c - np.dot(c[var_index], table[:, :-1])
        zs[-1] = -np.dot(c[var_index], table[:, -1])

        while True:
            in_index = -1
            for i in range(n):
                if zs[i] < 0:
                    in_index = i
                    break
            if in_index == -1:
                x = np.zeros(n)
                x[var_index] = table[:, -1]
                return x, -zs[-1]

            theta = np.inf
            out_index = -1
            for i in range(m):
                if table[i, in_index] > 0:
                    ratio = table[i, -1] / table[i, in_index]
                    if ratio < theta:
                        theta = ratio
                        out_index = i

            if out_index == -1:
                self.No_unbounded_solution += 1
                return None, None

            var_index[out_index] = in_index
            pivot = table[out_index, in_index]
            table[out_index, :] /= pivot

            for i in range(m):
                if i != out_index:
                    table[i, :] -= table[i, in_index] * table[out_index, :]

            zs[:-1] = c - np.dot(c[var_index], table[:, :-1])
            zs[-1] = -np.dot(c[var_index], table[:, -1])

    def solve(self, A, b, c):
        """
        完整求解单纯形法
        """
        try:
            c, A, b = self.convert_to_standard_form(c, A, b)
            A, b = self.remove_redundant_constraints(A, b)
            len_x = A.shape[1]
            A, c, initial_basic, M = self.initialize_feasible_solution(A, b, c)
            x, obj = self.simplex_method(A, b, c, initial_basic, M)
            if x is None or np.any(x[len_x:] > 1e-6):
                self.No_feasible_solution += 1
                return None, None
            return x[:len_x], obj
        except Exception as e:
            print(f"Error: {e}")
            return None, None


def main():
    print("Initializing data generator and simplex solver...")
    generator = DataGenerator()
    solver = SimplexSolver()

    print("Generating a random linear programming problem and saving to files...")
    A, b, c = generator.generate_random_input(m=50)
    generator.save_lp_problem(A, b, c, "LP_input.in", "LP_expected_output.out")
    print("Random LP problem saved to 'LP_input.in' and 'LP_expected_output.out'.")

    print("Solving the generated random linear programming problem...")
    x, obj = solver.solve(A, b, c)
    if x is not None:
        print(f"Solution found! Optimal x: {x}, Optimal objective value: {obj}")
    else:
        print("No feasible solution or the problem is unbounded!")

    print("Testing runtime for problems of different sizes...")
    ts = list(range(10, 201, 10))
    times = []
    for idx, m in enumerate(ts):
        print(f"Solving problem with m={m} ({idx + 1}/{len(ts)})...", end=" ")
        A, b, c = generator.generate_random_input(m)
        start_time = time.time()
        x, obj = solver.solve(A, b, c)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        if x is not None:
            print(f"Solved! Objective: {obj:.6f}, Time: {elapsed_time:.6f}s")
        else:
            print(f"No solution or unbounded! Time: {elapsed_time:.6f}s")

    print("Plotting runtime vs problem size...")
    plt.plot(ts, times)
    plt.xlabel("Number of variables (m)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Problem Size")
    plt.grid()
    plt.show()

    print(f"Test completed. Infeasible problems: {solver.No_feasible_solution}, Unbounded problems: {solver.No_unbounded_solution}")

if __name__ == "__main__":
    main()