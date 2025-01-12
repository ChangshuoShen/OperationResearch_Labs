import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import linprog
from ERGraph import Graph  # 导入 ERGraph 模块


class DijkstraSolver:
    def __init__(self, graph):
        """
        初始化 DijkstraSolver 类，传入图的邻接矩阵。
        """
        self.graph = graph

    def solve(self, start_node, end_node=None):
        """
        使用 Dijkstra 算法计算从 start_node 到所有节点的最短路径。
        如果指定 end_node，则提前退出。

        Returns:
            distances: 从 start_node 到所有节点的最短距离
            predecessor: 从 start_node 到每个节点的前驱数组
        """
        num_nodes = len(self.graph)
        distances = [float('inf')] * num_nodes
        predecessor = [-1] * num_nodes
        distances[start_node] = 0
        visited = [False] * num_nodes
        heap = []
        heapq.heappush(heap, (0, start_node))  # (距离, 节点)

        while heap:
            current_distance, current_node = heapq.heappop(heap)
            if visited[current_node]:
                continue
            visited[current_node] = True
            if end_node is not None and current_node == end_node:
                break
            for neighbor in range(num_nodes):
                weight = self.graph[current_node][neighbor]
                if weight > 0:  # 如果存在边
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        predecessor[neighbor] = current_node
                        heapq.heappush(heap, (distance, neighbor))
        return distances, predecessor

    @staticmethod
    def reconstruct_path(predecessor, start_node, end_node):
        """
        根据前驱数组重建从 start_node 到 end_node 的路径。
        """
        path = []
        current_node = end_node
        while current_node != -1:
            path.append(current_node)
            if current_node == start_node:
                break
            current_node = predecessor[current_node]
        path.reverse()
        return path if path[0] == start_node else []


def solve_lp(graph, start_node, end_node):
    """
    使用线性规划求解最短路径问题。

    Returns:
        result.fun: 最小路径长度
        path: 最短路径的节点序列
    """
    num_nodes = len(graph)
    num_edges = np.sum(graph > 0)
    edges = np.argwhere(graph > 0)
    costs = graph[graph > 0]

    A_eq = np.zeros((num_nodes, num_edges))
    b_eq = np.zeros(num_nodes)

    for i, (u, v) in enumerate(edges):
        A_eq[u, i] = 1
        A_eq[v, i] = -1

    b_eq[start_node] = 1
    b_eq[end_node] = -1
    bounds = [(0, 1) for _ in range(num_edges)]

    result = linprog(costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        x = result.x
        path = [start_node]
        current_node = start_node
        while current_node != end_node:
            for i, (u, v) in enumerate(edges):
                if u == current_node and x[i] > 0.5:
                    path.append(v)
                    current_node = v
                    break
        return result.fun, path
    else:
        raise ValueError("线性规划未能找到可行解。")


def main():
    num_nodes_list = list(range(10, 301, 10))  # 节点数量范围
    dijkstra_times = []
    lp_times = []
    dijkstra_vars = []
    lp_vars = []

    for num_nodes in num_nodes_list:
        print(f"Testing for graph size: {num_nodes} nodes")
        dijkstra_run_times = []
        lp_run_times = []

        for _ in tqdm(range(20), desc=f"Solving for {num_nodes} nodes", leave=True):
            # 生成随机连通图
            graph = Graph(graph_type='ER', N=num_nodes, p=0.1, max_len=10, plot_g=False)
            weight_matrix = graph.W

            # 初始化起点和终点
            start_node = 0
            end_node = num_nodes - 1

            # 测试 Dijkstra 算法
            dijkstra_solver = DijkstraSolver(weight_matrix)
            start_time = time.time()
            dijkstra_distances, dijkstra_predecessor = dijkstra_solver.solve(start_node, end_node)
            dijkstra_run_times.append(time.time() - start_time)

            # 测试线性规划方法
            start_time = time.time()
            _ = solve_lp(weight_matrix, start_node, end_node)
            lp_run_times.append(time.time() - start_time)

        # 计算运行时间的平均值和方差
        dijkstra_times.append(np.mean(dijkstra_run_times))
        dijkstra_vars.append(np.var(dijkstra_run_times))
        lp_times.append(np.mean(lp_run_times))
        lp_vars.append(np.var(lp_run_times))

    # 绘制结果并保存
    plt.figure(figsize=(10, 5))
    plt.plot(num_nodes_list, dijkstra_times, label="Dijkstra (Mean Runtime)", marker="o")
    plt.plot(num_nodes_list, lp_times, label="Linear Programming (Mean Runtime)", marker="x")
    plt.fill_between(num_nodes_list,
                     [d - np.sqrt(v) for d, v in zip(dijkstra_times, dijkstra_vars)],
                     [d + np.sqrt(v) for d, v in zip(dijkstra_times, dijkstra_vars)],
                     alpha=0.2, label="Dijkstra (Variance)")
    plt.fill_between(num_nodes_list,
                     [d - np.sqrt(v) for d, v in zip(lp_times, lp_vars)],
                     [d + np.sqrt(v) for d, v in zip(lp_times, lp_vars)],
                     alpha=0.2, label="Linear Programming (Variance)")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Runtime (s)")
    plt.title("Comparison of Dijkstra and Linear Programming")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("runtime_comparison.png")
    plt.show()

    print("Results plotted and saved as 'runtime_comparison.png'.")


if __name__ == '__main__':
    main()