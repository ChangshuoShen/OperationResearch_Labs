import numpy as np
import random
import time
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from tqdm import tqdm

class DataGenerator:
    """
    DataGenerator class to generate random linear programming problems.
    """
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def generate_random_input(self, m=None):
        """
        Generate random matrix A, vectors b and c.
        :param m: Optional number of variables m. If None, it is randomly chosen.
        :return: A, b, c, x (where x is the solution used to generate b)
        """
        if m is None:
            m = random.randint(10, 200)  # Randomly choose m in [10, 200]
        n = m - random.randint(0, m // 3)  # Ensure n <= m and m - n <= m / 3
        A = np.random.uniform(-10, 10, (n, m))
        x = np.random.uniform(0, 10, m)  # Generate solution x
        b = A @ x  # Ensure b = Ax
        c = np.random.uniform(-10, 10, m)  # Random coefficients for the objective
        return A, b, c, x


class SimplexSolver:
    """
    SimplexSolver class to solve linear programming problems using simplex method.
    """
    def __init__(self):
        self.No_feasible_solution = 0  # Count of infeasible problems
        self.No_unbounded_solution = 0  # Count of unbounded problems

    @staticmethod
    def convert_to_standard_form(c, A, b):
        """
        Convert to standard form by removing zero rows and flipping signs.
        """
        zero_row_indices = [i for i in range(len(b)) if np.all(A[i, :] == 0)]
        A = np.delete(A, zero_row_indices, axis=0)
        b = np.delete(b, zero_row_indices, axis=0)
        for i in range(A.shape[0]):
            if b[i] < 0:
                b[i] = -b[i]
                A[i] = -A[i]
        return c, A, b

    @staticmethod
    def remove_redundant_constraints(A, b):
        """
        Remove redundant constraints using QR decomposition.
        """
        A = np.column_stack((A, b))
        q, r = np.linalg.qr(A.T)
        valid_indices = np.abs(np.diag(r)) > 1e-6
        A_reduced = (A.T[:, valid_indices]).T
        return A_reduced[:, :-1], A_reduced[:, -1]

    @staticmethod
    def initialize_feasible_solution(A, b, c):
        """
        Use the Big-M method to initialize a feasible solution.
        """
        m, n = A.shape
        M = max((np.abs(c)).max() * 1e5, 1e6)
        artificial_vars = np.eye(m)
        A_aug = np.hstack([A, artificial_vars])
        c_aug = np.hstack([c, M * np.ones(m)])
        initial_basic = np.concatenate([np.zeros(n), b])
        return A_aug, c_aug, initial_basic, M

    def simplex_method(self, A, b, c, initial_basic, M):
        """
        Perform simplex method iterations.
        """
        m, n = A.shape
        x = initial_basic
        var_index = np.arange(n - m, n)
        table = np.column_stack((A, b))
        zs = np.zeros(n + 1)
        zs[:-1] = c - np.dot(c[var_index], table[:, :-1])
        zs[-1] = -np.dot(c[var_index], table[:, -1])

        while True:
            in_index = next((i for i in range(n) if zs[i] < 0), -1)
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
                raise ValueError("Linear programming problem is unbounded.")

            var_index[out_index] = in_index
            pivot = table[out_index, in_index]
            table[out_index, :] /= pivot
            for i in range(m):
                if i != out_index:
                    table[i, :] -= table[i, in_index] * table[out_index, :]
            zs[:-1] = c - np.dot(c[var_index], table[:, :-1])
            zs[-1] = -np.dot(c[var_index], table[:, -1])

        
    def solve(self, A, b, c, use_blands_rule=False):
        """
        Solve the linear programming problem using the simplex method.
        """
        try:
            c, A, b = self.convert_to_standard_form(c, A, b)
            A, b = self.remove_redundant_constraints(A, b)
            len_x = A.shape[1]
            A, c, initial_basic, M = self.initialize_feasible_solution(A, b, c)
            x, obj = self.simplex_method(A, b, c, initial_basic, M)
            # if use_blands_rule:
            #     x, obj = self.simplex_method_with_blands_rule(A, b, c, initial_basic, M)
            # else:
            #     x, obj = self.simplex_method(A, b, c, initial_basic, M)
            if np.any(np.abs(x[len_x:]) > 1e-6):
                self.No_feasible_solution += 1
                raise ValueError("Linear programming problem is infeasible.")
            return x[:len_x], obj
        
        except Exception as e:
            print(f"Error: {e}")
            return None, None

def main():
    print("Initializing data generator and simplex solver...")
    generator = DataGenerator()
    solver = SimplexSolver()

    print("Solving predefined test cases...")
    # 测试案例 1：有可行解
    print("\nTest Case 1: Feasible solution")
    A = np.array([
            [1, 2, 2, 1, 0, 0], 
            [2, 1, 2, 0, 1, 0],
            [2, 2, 1, 0, 0, 1]
        ])
    b = np.array([20, 20, 20])
    c = np.array([-10, -12, -12, 0, 0, 0])
    print("Matrix A:")
    print(A)
    print("Vector b:", b)
    print("Vector c:", c)
    simplex_solution, simplex_obj = solver.solve(A, b, c)
    print("Simplex Method Solution:", simplex_solution)
    print("Simplex Method Objective:", simplex_obj)
    linprog_res = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * len(c), method="highs")
    print("SciPy Linprog Solution:", linprog_res.x if linprog_res.success else "Failed")
    print("SciPy Linprog Objective:", linprog_res.fun if linprog_res.success else "Failed")

    # 测试案例 2：有冗余秩
    print("\nTest Case 2: Redundant constraints")
    A = np.array([
            [1, 2, 3], 
            [2, 4, 6],
            [1, 1, 1]
        ])
    b = np.array([6, 12, 3])
    c = np.array([1, 2, 3])
    print("Matrix A:")
    print(A)
    print("Vector b:", b)
    print("Vector c:", c)
    simplex_solution, simplex_obj = solver.solve(A, b, c)
    print("Simplex Method Solution:", simplex_solution)
    print("Simplex Method Objective:", simplex_obj)
    linprog_res = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * len(c), method="highs")
    print("SciPy Linprog Solution:", linprog_res.x if linprog_res.success else "Failed")
    print("SciPy Linprog Objective:", linprog_res.fun if linprog_res.success else "Failed")

    # 测试案例 3：无解（无可行域）
    print("\nTest Case 3: Infeasible solution (no feasible region)")
    A = np.array([
            [1, 0, 0], 
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1]
        ])
    b = np.array([2, 2, 2, 2])
    c = np.array([1, 1, 1])
    print("Matrix A:")
    print(A)
    print("Vector b:", b)
    print("Vector c:", c)
    simplex_solution, simplex_obj = solver.solve(A, b, c)
    print("Simplex Method Solution:", simplex_solution)
    print("Simplex Method Objective:", simplex_obj)
    linprog_res = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * len(c), method="highs")
    print("SciPy Linprog Solution:", linprog_res.x if linprog_res.success else "Failed")
    print("SciPy Linprog Objective:", linprog_res.fun if linprog_res.success else "Failed")

    # 测试案例 4：无解（无界）
    print("\nTest Case 4: Unbounded solution")
    A = np.array([
            [1, -1],
            [-1, 1]
        ])
    b = np.array([0, 0])
    c = np.array([-1, 0])
    print("Matrix A:")
    print(A)
    print("Vector b:", b)
    print("Vector c:", c)
    simplex_solution, simplex_obj = solver.solve(A, b, c)
    print("Simplex Method Solution:", simplex_solution)
    print("Simplex Method Objective:", simplex_obj)
    linprog_res = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * len(c), method="highs")
    print("SciPy Linprog Solution:", linprog_res.x if linprog_res.success else "Failed")
    print("SciPy Linprog Objective:", linprog_res.fun if linprog_res.success else "Failed")

    print("\nGenerating and solving random linear programming problems...")
    
    ts = list(range(10, 201, 10))  # 外层循环的变量数量范围
    simplex_times = []
    linprog_times = []
    simplex_vars = []
    linprog_vars = []

    for m in ts:  # 外层循环不使用 tqdm
        num_success = 0
        print(f"Testing for problem size m={m}")  # 显示当前外层循环进度
        simplex_run_times = []
        linprog_run_times = []

        # 内层循环使用 tqdm，显示 20 次测试的进度
        for _ in tqdm(range(20), desc=f"Solving for m={m}", leave=True):
            A, b, c, _ = generator.generate_random_input(m)

            # 测量单纯形法的运行时间
            start = time.time()
            simplex_solution, simplex_obj = solver.solve(A, b, c)
            simplex_run_times.append(time.time() - start)

            # 测量 SciPy linprog 的运行时间
            start = time.time()
            bounds = [(0, None)] * len(c)  # 修复 bounds 的长度
            linprog_res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
            linprog_run_times.append(time.time() - start)

            # 对比单纯形法和 linprog 的结果
            if linprog_res.success:
                linprog_obj = linprog_res.fun
                if not np.isclose(simplex_obj, linprog_obj, atol=1e-6):
                    print(f"Discrepancy detected: Simplex={simplex_obj}, Linprog={linprog_obj}")
                num_success += 1
            else:
                print("Linprog failed to find a solution.")
                
        print(f"Number of successes: {num_success} / 20")
        # 计算运行时间的平均值和方差
        simplex_times.append(np.mean(simplex_run_times))
        simplex_vars.append(np.var(simplex_run_times))
        linprog_times.append(np.mean(linprog_run_times))
        linprog_vars.append(np.var(linprog_run_times))

    # 绘制结果并保存
    print("Plotting results and saving figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 运行时间图
    axes[0].plot(ts, simplex_times, label="Simplex Method", marker="o")
    axes[0].plot(ts, linprog_times, label="SciPy Linprog", marker="x")
    axes[0].set_xlabel("Number of variables (m)")
    axes[0].set_ylabel("Average Runtime (s)")
    axes[0].set_title("Runtime vs Problem Size")
    axes[0].legend()
    axes[0].grid()

    # 方差图
    axes[1].plot(ts, simplex_vars, label="Simplex Method Variance", marker="o")
    axes[1].plot(ts, linprog_vars, label="SciPy Linprog Variance", marker="x")
    axes[1].set_xlabel("Number of variables (m)")
    axes[1].set_ylabel("Variance of Runtime")
    axes[1].set_title("Variance vs Problem Size")
    axes[1].legend()
    axes[1].grid()

    # 保存图像
    plt.tight_layout()
    plt.savefig("runtime_and_variance1.png")
    print("Figure saved as 'runtime_and_variance1.png'.")

if __name__ == "__main__":
    main()