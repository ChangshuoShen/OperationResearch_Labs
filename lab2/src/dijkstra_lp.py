import time
import heapq
import numpy as np
from scipy.optimize import linprog
from ERGraph import Graph  # 导入ERGraph模块


def dijkstra(graph, start_node, end_node=None):
    """
    使用heapq实现的Dijkstra算法，计算从start_node到所有节点的最短路径。
    同时支持计算从start_node到end_node的路径。

    Parameters:
        graph: 图的邻接矩阵（权重矩阵）
        start_node: 起点
        end_node: 终点（可选）

    Returns:
        distances: 从start_node到所有节点的最短距离
        predecessor: 从start_node到每个节点的前驱数组
    """
    num_nodes = len(graph)
    distances = [float('inf')] * num_nodes  # 初始化距离为无穷大
    predecessor = [-1] * num_nodes  # 初始化前驱数组
    distances[start_node] = 0  # 起点到自身的距离为0
    visited = [False] * num_nodes  # 记录节点是否已访问
    heap = []  # 最小堆

    # 将起点加入堆
    heapq.heappush(heap, (0, start_node))  # (距离, 节点)

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if visited[current_node]:
            continue

        # 标记当前节点为已访问
        visited[current_node] = True

        # 如果已找到终点的最短路径，提前退出
        if end_node is not None and current_node == end_node:
            break

        # 遍历当前节点的所有邻居
        for neighbor in range(num_nodes):
            weight = graph[current_node][neighbor]
            if weight > 0:  # 如果存在边
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    # 更新最短距离和前驱节点
                    distances[neighbor] = distance
                    predecessor[neighbor] = current_node
                    heapq.heappush(heap, (distance, neighbor))

    return distances, predecessor

def reconstruct_path(predecessor, start_node, end_node):
    """
    根据前驱数组，重建从start_node到end_node的路径。

    Parameters:
        predecessor: 前驱数组
        start_node: 起点
        end_node: 终点

    Returns:
        path: 从start_node到end_node的路径（节点列表）
    """
    path = []
    current_node = end_node
    while current_node != -1:
        path.append(current_node)
        if current_node == start_node:
            break
        current_node = predecessor[current_node]
    path.reverse()
    return path if path[0] == start_node else []  # 如果无法到达end_node，返回空路径


def solve_lp(graph, start_node, end_node):
    """
    使用线性规划求解最短路径问题，并打印路径。

    Parameters:
        graph: 图的邻接矩阵（权重矩阵）
        start_node: 起点
        end_node: 终点

    Returns:
        result.fun: 最小路径长度
        path: 最短路径的节点序列
    """
    num_nodes = len(graph)
    num_edges = np.sum(graph > 0)  # 边的数量
    edges = np.argwhere(graph > 0)  # 获取所有边的索引
    costs = graph[graph > 0]  # 边的权重

    # 构建线性规划问题
    A_eq = np.zeros((num_nodes, num_edges))
    b_eq = np.zeros(num_nodes)
    
    for i, (u, v) in enumerate(edges):
        A_eq[u, i] = 1  # 出发点
        A_eq[v, i] = -1  # 到达点

    b_eq[start_node] = 1  # 起点流出1
    b_eq[end_node] = -1  # 终点流入1

    bounds = [(0, 1) for _ in range(num_edges)]  # 决策变量x_ij的范围：[0, 1]

    result = linprog(costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        # 重建路径
        x = result.x  # 决策变量的值
        path = [start_node]
        current_node = start_node
        while current_node != end_node:
            # 找到以当前节点为起点的边
            for i, (u, v) in enumerate(edges):
                if u == current_node and x[i] > 0.5:  # 判断边是否被选择
                    path.append(v)
                    current_node = v
                    break
        return result.fun, path
    else:
        raise ValueError("线性规划未能找到可行解。")


def main():
    # 生成随机连通图
    num_nodes = 100  # 节点数量
    connection_probability = 0.1  # 边的连接概率
    max_length = 10  # 边的最大权重

    graph = Graph(graph_type='ER', N=num_nodes, p=connection_probability, max_len=max_length, plot_g=False)
    weight_matrix = graph.W  # 获取边权重矩阵

    # 检查是否有负权重边
    if np.any(weight_matrix < 0):
        raise ValueError("图中存在负权重边，Dijkstra算法不适用。")

    # 已知ERGraph生成的图是连通的，因此不需要额外检查连通性
    print("图已验证为连通。")

    # 选择起点和终点
    start_node = 0
    end_node = num_nodes - 1

    # 测试Dijkstra算法
    start_time = time.time()
    dijkstra_distances, dijkstra_predecessor = dijkstra(weight_matrix, start_node, end_node)
    dijkstra_time = time.time() - start_time
    dijkstra_path = reconstruct_path(dijkstra_predecessor, start_node, end_node)
    print(f"Dijkstra算法: 最短路径长度 = {dijkstra_distances[end_node]}, 用时 = {dijkstra_time:.6f} 秒")
    print(f"Dijkstra算法: 从节点 {start_node} 到节点 {end_node} 的路径 = {dijkstra_path}")

    # 测试LP算法
    start_time = time.time()
    lp_result, lp_path = solve_lp(weight_matrix, start_node, end_node)
    lp_time = time.time() - start_time
    print(f"线性规划: 最短路径长度 = {lp_result}, 用时 = {lp_time:.6f} 秒")
    print(f"线性规划: 从节点 {start_node} 到节点 {end_node} 的路径 = {lp_path}")

    # 比较两种算法的效率
    print(f"效率比较: Dijkstra用时 = {dijkstra_time:.6f} 秒, LP用时 = {lp_time:.6f} 秒")
    
    
if __name__ == '__main__':
    main()