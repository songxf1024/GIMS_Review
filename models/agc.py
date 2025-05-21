import random
import time

import cv2
import dgl
import numpy as np
import torch
from collections import deque

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
from scipy.spatial import cKDTree, Delaunay
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity


class Timer:
    """
    一个计时上下文管理器，用于测量代码块的执行时间。
    它可以根据用户的需求以秒或纳秒为单位测量时间。

    属性:
        use_ns (bool): 如果为True，则计时器以纳秒为单位测量时间；如果为False，则以秒为单位。
        start (float|int): 测量开始的时间。
        end (float|int): 测量结束的时间。
        interval (float|int): 计算的开始和结束时间之间的持续时间。
    """

    def __init__(self, note='', use_ns=False):
        """
        使用选择是否使用纳秒精度初始化 Timer。

        参数:
            use_ns (bool): 确定是否使用纳秒进行时间测量，默认为False。
        """
        self.use_ns = use_ns
        self.start = None
        self.end = None
        self.interval = None
        self.note = note

    def __enter__(self):
        """
        启动计时器。当进入上下文块时记录开始时间。

        返回:
            Timer: 返回自身对象，以便在上下文外部访问属性。
        """
        self.start = time.perf_counter_ns() if self.use_ns else time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        结束计时器。当退出上下文块时记录结束时间。
        此函数还计算时间间隔并打印经过的时间。

        参数:
            exc_type: 如果在上下文中引发异常，则为异常类型。
            exc_value: 如果引发异常，则为异常值。
            traceback: 如果发生异常，则为回溯详细信息。

        返回:
            None
        """
        self.end = time.perf_counter_ns() if self.use_ns else time.perf_counter()
        self.interval = self.end - self.start
        print((f'{self.note} ' if self.note else '') + f"耗时：{self.interval:.6f} 秒")


#-------------------------------------------------------------#

def build_dgl_graph_supernode(super_nodes, device):
    # 假设super_nodes是单个图的超节点列表
    g = dgl.DGLGraph()
    g.add_nodes(len(super_nodes))

    # 添加节点特征
    node_features = torch.stack([node['descriptor'] for node in super_nodes]).to(device)
    g.ndata['feat'] = node_features

    # 添加边
    src, dst = [], []
    for super_node in super_nodes:
        for edge_target in super_node['edges']:
            src.append(super_node['id'])
            dst.append(edge_target)
    g.add_edges(src, dst)

    return g

def create_supernodes_batched2(batched_graph, n, strategy='max_neighbors', exclude_used_nodes=True):
    batched_supernodes = []

    for graph in batched_graph:
        supernodes = []
        supernode_edges = {}
        visited = set() if exclude_used_nodes else None

        def local_search(start_node_index):
            if visited is not None and start_node_index in visited:
                return None, None
            supernode_id = len(supernodes)
            queue = deque([start_node_index])
            supernode = {'id': supernode_id, 'nodes': [start_node_index], 'edges': []}
            if visited is not None:
                visited.add(start_node_index)

            descriptors = [graph[start_node_index]['descriptor']]
            while queue and len(supernode['nodes']) < n:
                current_index = queue.popleft()
                for neighbor_index in graph[current_index]['edges']:
                    if visited is None or neighbor_index not in visited:
                        supernode['nodes'].append(neighbor_index)
                        queue.append(neighbor_index)
                        if visited is not None:
                            visited.add(neighbor_index)
                        descriptors.append(graph[neighbor_index]['descriptor'])
                        for sn_id, sn_nodes in supernode_edges.items():
                            if neighbor_index in sn_nodes and supernode_id != sn_id:
                                supernode['edges'].append(sn_id)
                                break

            # 使用PyTorch计算平均得分和描述符
            scores_tensor = torch.stack([graph[node]['score'].clone().detach() for node in supernode['nodes']])
            supernode['score'] = scores_tensor.mean().item()
            # 直接使用起始节点的坐标作为超节点的中心坐标
            supernode['center'] = graph[start_node_index]['point'].clone().detach().cpu().numpy()

            if descriptors:
                supernode['descriptor'] = torch.stack(descriptors).mean(dim=0).cpu().numpy()
            else:
                supernode['descriptor'] = torch.zeros(128, device=graph[0]['descriptor'].device).cpu().numpy()
            supernode_edges[supernode_id] = supernode['nodes']
            return supernode, supernode_edges

        for i in range(len(graph)):
            if visited is None or i not in visited:
                new_supernode, _ = local_search(i)
                if new_supernode:
                    supernodes.append(new_supernode)
        for supernode in supernodes:
            supernode['edges'] = list(set(supernode['edges']))
        batched_supernodes.append(supernodes)
    return batched_supernodes

# 聚类并生成超节点
def create_supernodes_batched_cluster(batched_graph, n_clusters=None):
    # 处理每一层，生成超节点
    super_graphs = []

    for graph in batched_graph:
        descriptors = np.array([node['descriptor'].cpu().numpy() for node in graph])
        n_clusters = n_clusters or int(len(graph) / 4)
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit(descriptors)
        labels = kmeans.labels_
        super_nodes = []
        node_id_to_super_node = {}

        for cluster_id in range(n_clusters):
            cluster_nodes = [node for node, label in zip(graph, labels) if label == cluster_id]
            if not cluster_nodes:
                continue
            # 选择得分最高的节点作为center
            center_node = max(cluster_nodes, key=lambda node: node['score'])
            avg_score = torch.stack([node['score'] for node in cluster_nodes]).mean()
            avg_descriptor = torch.stack([node['descriptor'] for node in cluster_nodes]).mean(dim=0)

            super_node = {
                'id': cluster_id,
                'nodes': [node['id'] for node in cluster_nodes],
                'edges': [],  # 将在后续步骤中填充
                'score': avg_score,
                'center': center_node['point'],
                'descriptor': avg_descriptor,
            }

            for node in cluster_nodes:
                node_id_to_super_node[node['id']] = super_node['id']
            super_nodes.append(super_node)

        # 处理超节点间的边
        for super_node in super_nodes:
            edge_super_nodes = set()
            for node_id in super_node['nodes']:
                original_node = next((node for node in graph if node['id'] == node_id), None)
                if original_node:
                    for edge in original_node['edges']:
                        edge_super_node_id = node_id_to_super_node.get(edge)
                        if edge_super_node_id and edge_super_node_id != super_node['id']:
                            edge_super_nodes.add(edge_super_node_id)
            super_node['edges'] = list(edge_super_nodes)
        super_graphs.append(super_nodes)
    return super_graphs

def create_supernodes_batched(batched_graph, radius):
    super_graphs = []

    for graph in batched_graph:
        points = np.array([node['point'].cpu().numpy() for node in graph])
        # 计算所有节点之间的欧式距离
        distances = euclidean_distances(points, points)
        # 根据半径确定哪些节点应该被连接
        adjacency = distances < radius

        super_nodes = []
        node_id_to_super_node = {}

        for idx, node in enumerate(graph):
            connected_nodes = np.where(adjacency[idx])[0]
            if len(connected_nodes) > 0:
                avg_descriptor = torch.stack([graph[i]['descriptor'] for i in connected_nodes]).mean(dim=0)

                super_node = {
                    'id': idx,
                    'nodes': connected_nodes.tolist(),
                    'edges': [],
                    'score': node['score'],
                    'point': node['point'],
                    'descriptor': avg_descriptor,
                }

                for connected_node_id in connected_nodes:
                    node_id_to_super_node[connected_node_id] = idx
                super_nodes.append(super_node)

        # 处理超节点间的边
        for super_node in super_nodes:
            edge_super_nodes = set()
            for node_id in super_node['nodes']:
                original_node = graph[node_id]
                for edge in original_node['edges']:
                    edge_super_node_id = node_id_to_super_node.get(edge)
                    if edge_super_node_id and edge_super_node_id != super_node['id']:
                        edge_super_nodes.add(edge_super_node_id)
            super_node['edges'] = list(edge_super_nodes)

        super_graphs.append(super_nodes)
    return super_graphs

def process_batched_supernodes(supernodes_batched, device):
    keypoints_list = []
    descriptors_list = []
    scores_list = []
    for supernodes in supernodes_batched:
        keypoints_tensor = torch.stack([s['point'] for s in supernodes]).float()
        descriptors_tensor = torch.stack([s['descriptor'] for s in supernodes]).float()
        scores_tensor = torch.stack([s['score'] for s in supernodes]).float()
        keypoints_list.append(keypoints_tensor)
        descriptors_list.append(descriptors_tensor)
        scores_list.append(scores_tensor)

    keypoints_batched = torch.stack(keypoints_list).to(device)
    descriptors_batched = torch.stack(descriptors_list).to(device)
    scores_batched = torch.stack(scores_list).to(device)
    descriptors_batched = descriptors_batched.permute(0, 2, 1)
    return keypoints_batched, descriptors_batched, scores_batched

#-------------------------------------------------------------#

def visualize_graph_with_coordinates(graph,
                                     show_node_scores=False,
                                     node_size_range=(3, 300),
                                     edge_alpha=0.8,
                                     edge_width=1,
                                     figsize=(8, 8),
                                     title="Graph Visualization",
                                     image=None):
    """
    可视化带点坐标的 NetworkX 图（节点位置从 node["point"] 获取）。

    参数:
        graph (nx.Graph): 包含 point/feat/score 属性的图
        show_node_scores (bool): 是否用得分映射节点颜色
        node_size_range (tuple): 最小与最大节点大小
        edge_alpha (float): 边透明度
        figsize (tuple): 图像大小
        title (str): 标题
        image (np.ndarray): 若提供则作为图像背景（HxW或HxWx3）
    """
    pos = {i: data['point'][:2] for i, data in graph.nodes(data=True)}  # 只取前两个维度（xy）
    scores = np.array([data['score'] for _, data in graph.nodes(data=True)])

    # 节点颜色与大小
    if show_node_scores:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        node_sizes = norm_scores * (node_size_range[1] - node_size_range[0]) + node_size_range[0]
        node_colors = norm_scores
    else:
        node_sizes = node_size_range[0]
        node_colors = "blue"
    plt.figure(figsize=figsize)
    # 绘制背景图像（如果有）
    if image is not None:
        if image.ndim == 2:  # 灰度图
            plt.imshow(image, cmap='gray', origin='upper')
        else:  # 彩色图
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), origin='upper')
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color="red",
        width=edge_width,
        alpha=edge_alpha,
        cmap=plt.cm.viridis,
    )
    plt.title(title)
    plt.axis("equal")
    plt.show()

def create_graph_from_keypoints(keypoints, descriptors, scores, radius, device=torch.device('cpu')):
    def calc_distances_numpy(points):
        return np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2))
    def calc_distances_torch(points):
        return torch.sqrt(torch.sum((points.unsqueeze(1) - points.unsqueeze(0)) ** 2, dim=2))
    batched_graph = []
    for batched_idx in range(keypoints.shape[0]):
        kpts, descs, scrs = keypoints[batched_idx], descriptors[batched_idx].permute(1, 0), scores[batched_idx]
        dist_matrix = calc_distances_numpy(kpts) if isinstance(kpts, np.ndarray) else calc_distances_torch(kpts).cpu().numpy()
        edges = dist_matrix < radius
        graph = []
        for i in range(kpts.shape[0]):
            kpt, desc, scr = kpts[i], descs[i], scrs[i]
            connected_nodes = np.where(edges[i])[0].tolist()
            # Ensure there are no isolated nodes
            if len(connected_nodes) == 1:  # Only connected to itself
                # Find the nearest node (excluding itself) and connect
                nearest_node = np.argsort(dist_matrix[i])[1]  # [0] would be the node itself
                connected_nodes.append(nearest_node)

            graph.append({
                'id': i,
                'point': kpt,
                'score': scr,
                'descriptor': desc,
                'edges': connected_nodes,
            })

        batched_graph.append(graph)
        # batched_graph.append(remove_small_components(graph))
    return batched_graph

def create_dgl_graph_from_batched_graph(batched_graph, device=torch.device('cpu')):
    dgl_graphs = []

    for graph in batched_graph:
        # 创建一个空的DGL图
        g = dgl.graph(([], []), num_nodes=len(graph), device=device)

        # 提取并设置节点特征
        g.ndata['feat'] = torch.stack([node['descriptor'] for node in graph]).to(device)
        g.ndata['point'] = torch.stack([node['point'] for node in graph]).to(device)
        g.ndata['score'] = torch.stack([node['score'] for node in graph]).to(device)

        # 提取并添加边缘
        edges_src = []
        edges_dst = []
        for node in graph:
            for neighbor in node['edges']:
                edges_src.append(node['id'])
                edges_dst.append(neighbor)
        src_tensor = torch.tensor(edges_src, dtype=torch.long, device=device)
        dst_tensor = torch.tensor(edges_dst, dtype=torch.long, device=device)
        g.add_edges(src_tensor, dst_tensor)

        # 保存构建的DGL图
        dgl_graphs.append(g)

    return dgl_graphs

def fast_percentile_threshold(similarities, percentile):
    """
    快速近似分位数计算
    对于大规模数据，这种方法比 np.percentile 更快，尤其适用于图构建中
    对边相似度的分位数筛选。

    参数:
        similarities (np.ndarray 或 torch.Tensor): 相似度向量（一维）。
        percentile (float): 分位数（0~100），如 50 表示中位数。

    返回:
        float: 近似的分位数值。
    """
    k = int(len(similarities) * percentile / 100)
    if k >= len(similarities): k = len(similarities) - 1
    return np.partition(similarities, k)[k]

def fast_cosine_similarity_matrix(descs: np.ndarray) -> np.ndarray:
    """
    使用 PyTorch 快速计算余弦相似度矩阵。
    输入： descs: np.ndarray of shape (N, D)，未归一化的特征向量
    输出： sim_matrix: np.ndarray of shape (N, N)，归一化后的余弦相似度矩阵
    """
    descs = torch.from_numpy(descs).float()  # 转为 float32 tensor
    descs = torch.nn.functional.normalize(descs, dim=1)  # L2 归一化
    sim_matrix = torch.matmul(descs, descs.T)  # 点积即余弦相似度
    return sim_matrix.cpu().numpy()

def calculate_cosine_similarity_threshold(descriptors, percentile=50, similarity_matrix=None):
    """
    根据一批描述符的两两余弦相似度，计算指定百分位的动态阈值。
    为避免构建过于稠密的图，本函数可以自适应地计算一个余弦相似度的阈值，
    仅保留高于该阈值的边，用于后续图构建。

    参数:
        descriptors (np.ndarray): shape=(N, D)，关键点的描述符向量（未归一化）。
        percentile (float):      用于设定阈值的分位数（例如 50 表示中位数）。
        similarity_matrix (np.ndarray|None): 可选，预先计算好的相似度矩阵，若为 None 则会自动计算。

    返回:
        float: 根据指定 percentile 计算出的相似度阈值。
    """
    # similarity_matrix = cosine_similarity(descriptors) if similarity_matrix is None else similarity_matrix
    similarity_matrix = fast_cosine_similarity_matrix(descriptors) if similarity_matrix is None else similarity_matrix
    similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    similarities = similarities[similarities > 0]
    if len(similarities) == 0: return 0.0  # fallback 值，避免崩溃
    # similarity_threshold = np.percentile(similarities, percentile)
    similarity_threshold = fast_percentile_threshold(similarities, percentile)
    return similarity_threshold

def fast_build_graph_with_cosine_similarity(keypoints, descriptors, scores, radius, percentile=50):
    """
    构建带余弦相似度过滤的空间图（支持动态阈值、快速相似度计算、KDTree批量邻接）。

    参数:
        keypoints (torch.Tensor): shape (B, N, 2 or 3)，每批关键点坐标
        descriptors (torch.Tensor): shape (B, D, N)，每批关键点描述符（未归一化）
        scores (torch.Tensor): shape (B, N)，每批关键点得分
        radius (float): 邻接半径阈值
        percentile (float): 用于过滤的余弦相似度百分位阈值

    返回:
        List[nx.Graph]: 每批图对应的 NetworkX 表达形式
    """
    batch_graphs = []
    B = keypoints.shape[0]
    for b in range(B):
        # === Step 1: 提取每一图的节点数据 ===
        kpts = keypoints[b].cpu().numpy()  # shape (N, 2/3)
        descs = descriptors[b].permute(1, 0).cpu().numpy()  # shape (N, D)
        scrs = scores[b].cpu().numpy()  # shape (N,)
        N = len(kpts)
        # === Step 2: 建立 KDTree 并获取所有 radius 内的点对 (i,j)，i<j ===
        kdtree = cKDTree(kpts)
        edge_candidates = kdtree.query_pairs(r=radius)  # set of (i, j)
        # === Step 3: 计算余弦相似度矩阵（使用你已有的优化版本） ===
        sim_matrix = fast_cosine_similarity_matrix(descs)  # (N, N)
        similarity_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        threshold = fast_percentile_threshold(similarity_values, percentile)
        # === Step 4: 构建图 ===
        graph = nx.Graph()
        for i in range(N):
            graph.add_node(i, point=kpts[i], feat=descs[i], score=scrs[i])
        for i, j in edge_candidates:
            if sim_matrix[i, j] >= threshold:
                graph.add_edge(i, j)
        batch_graphs.append(graph)
    return batch_graphs

def build_graph_with_cosine_similarity(keypoints, descriptors, scores, radius, percentile=50):
    keypoints = keypoints.cpu().numpy()
    descriptors = descriptors.permute(0, 2, 1).cpu().numpy() # => (batch, number, dim)
    scores = scores.cpu().numpy()
    batch_graph = []
    for batched_idx in range(keypoints.shape[0]):
        kpts = keypoints[batched_idx]
        descs = descriptors[batched_idx]
        scrs = scores[batched_idx]
        kdtree = cKDTree(kpts)
        graph = nx.Graph()
        # similarity_matrix = cosine_similarity(descs)
        similarity_matrix = fast_cosine_similarity_matrix(descs)
        similarity_threshold = calculate_cosine_similarity_threshold(descs, percentile)
        # 创建节点
        for i in range(len(kpts)): graph.add_node(i, point=kpts[i], feat=descs[i], score=scrs[i])
        # 通过空间和描述符相似度添加边
        for i, kpt in enumerate(kpts):
            neighbors_idx = kdtree.query_ball_point(kpt, r=radius)
            for j in neighbors_idx:
                if i != j and similarity_matrix[i, j] >= similarity_threshold:
                    graph.add_edge(i, j)
        batch_graph.append(graph)
    return batch_graph

def connect_isolated_nodes(graph):
    """
    Connect isolated nodes in the graph to their nearest neighbor, using positions stored in node attributes.

    Parameters:
    - graph: networkx.Graph, the graph to process

    Returns:
    - graph: networkx.Graph, the graph with isolated nodes connected
    """
    if len(graph.nodes) == 0 or len(graph.edges) == 0: return graph
    positions = np.array([graph.nodes[node]['point'] for node in graph.nodes])
    kdtree = cKDTree(positions)
    for node in graph.nodes:
        if graph.degree(node) == 0:
            _, nearest_neighbor = kdtree.query(positions[node], k=2)
            nearest_neighbor_index = nearest_neighbor[1] if nearest_neighbor[0] == node else nearest_neighbor[0]
            nearest_neighbor_id = list(graph.nodes)[nearest_neighbor_index]
            graph.add_edge(node, nearest_neighbor_id)
    return graph

def remove_small_components(graph, min_size=5):
    """
    Remove small components from the graph.

    Parameters:
    - graph: networkx.Graph, the graph from which small components will be removed
    - min_size: int, the minimum size of components to be retained in the graph

    Returns:
    - cleaned_graph: networkx.Graph, the graph with small components removed
    """
    cleaned_graph = graph.copy()
    kept_nodes = set()
    for component in list(nx.connected_components(graph)):
        if len(component) < min_size:
            for node in component:
                cleaned_graph.remove_node(node)
        else:
            kept_nodes.update(component)
    return cleaned_graph, kept_nodes

def fast_connect_components(graph):
    """
    使用改进的KDTree方法快速连接图中的所有连通分量。
    与 connect_components 相比，该方法在每轮中让每个分量都连接到其最近的分量，
    极大减少了连接轮数，提高了效率，适用于大图或分量较多的情况。
    改进策略：
    - 为每个分量计算其质心。
    - 使用 KDTree 快速找到每个分量最近的其他分量。
    - 对每对分量，寻找它们之间最近的节点对，并添加边连接它们。
    Parameters: graph (networkx.Graph): 输入的图，图中每个节点需包含 'point' 属性，表示空间坐标。
    Returns: networkx.Graph: 所有分量已通过最近点连接后的图。
    注意事项:
        - 若图本身已连通，将不做修改。
        - 若图中节点数量较小或只有一个分量，性能收益不明显。
        - 不保证构成全局最短连接，仅为快速启发式连接方案。
    """
    components = list(nx.connected_components(graph))
    if len(components) <= 1:
        return graph
    # 为每个组件计算中心坐标
    centroids = []
    print(f'>> Total Components to Refine: {len(components)}')
    for comp in components:
        positions = np.array([graph.nodes[n]['point'] for n in comp])
        centroid = positions.mean(axis=0)
        centroids.append(centroid)
    # 构建 KDTree 搜索最近组件
    tree = cKDTree(centroids)
    _, nn_indices = tree.query(centroids, k=2)  # 最近邻（跳过自己）
    connected = set()
    for i, j in enumerate(nn_indices[:, 1]):
        if (i, j) in connected or (j, i) in connected:
            continue
        connected.add((i, j))
        comp_i = list(components[i])
        comp_j = list(components[j])
        # 找 comp_i 与 comp_j 之间最近的点对
        points_i = np.array([graph.nodes[u]['point'] for u in comp_i])
        points_j = np.array([graph.nodes[v]['point'] for v in comp_j])
        tree_i = cKDTree(points_i)
        dists, indices = tree_i.query(points_j, k=1)
        idx_j = np.argmin(dists)
        idx_i = indices[idx_j]
        u = comp_i[idx_i]
        v = comp_j[idx_j]
        graph.add_edge(u, v)
    return graph

def connect_components(graph):
    """
    使用KD-Tree快速连接图中的各个分量。
    connect_components的优化加速版本。构图效果略有不同，但加速非常明显。
    """
    while nx.number_connected_components(graph) > 1:
        components = list(nx.connected_components(graph))
        component_positions = []
        for component in components:
            positions = np.array([graph.nodes[node]['point'] for node in component])
            centroid = np.mean(positions, axis=0)
            component_positions.append(centroid)
        tree = cKDTree(component_positions)
        for i, pos in enumerate(component_positions):
            _, j = tree.query(pos, k=2)
            closest_pair = (i, j[1])
            break
        # 获取最近分量的节点
        component1, component2 = components[closest_pair[0]], components[closest_pair[1]]
        positions1 = np.array([graph.nodes[node]['point'] for node in component1])
        positions2 = np.array([graph.nodes[node]['point'] for node in component2])
        tree1 = cKDTree(positions1)
        dists, indices = tree1.query(positions2, k=1)
        # 确定最小距离的索引
        idx2 = np.argmin(dists)  # 最小距离的位置2的索引
        idx1 = indices[idx2]  # 从位置1的索引数组中取出对应的索引
        # 从原始组件列表中选择节点
        node1 = list(component1)[idx1]
        node2 = list(component2)[idx2]
        closest_nodes = (node1, node2)
        # 添加连接最近节点的边
        graph.add_edge(*closest_nodes)
    return graph

def connect_components_mst(graph):
    """
    使用最小生成树（MST）的方法连接图中的各个分量。
     connect_components的优化加速版本。构图效果一样，但加速非常明显。
    """
    # 获取所有分量
    components = list(nx.connected_components(graph))
    component_nodes = [list(comp) for comp in components]
    num_components = len(components)
    if num_components == 1:
        return graph  # 如果已经连通，直接返回
    # 构建一个完全图来表示各个分量之间的最短距离
    distances = np.zeros((num_components, num_components))
    distances.fill(np.inf)
    # 为每个分量计算质心
    centroids = []
    for nodes in component_nodes:
        positions = np.array([graph.nodes[node]['point'] for node in nodes])
        centroid = np.mean(positions, axis=0)
        centroids.append(centroid)
    # 使用KD-Tree优化距离计算
    tree = cKDTree(centroids)
    for i in range(num_components):
        for j in range(i + 1, num_components):
            if distances[i, j] == np.inf:  # 避免重复计算
                pos1 = np.array([graph.nodes[node]['point'] for node in component_nodes[i]])
                pos2 = np.array([graph.nodes[node]['point'] for node in component_nodes[j]])
                tree1 = cKDTree(pos1)
                dists, idxs = tree1.query(pos2, k=1)
                idx1 = idxs[np.argmin(dists)]  # 对多个距离取最小值对应的idx1
                idx2 = np.argmin(dists)  # 对多个距离取最小值的索引idx2
                min_distance = dists.min()
                distances[i, j] = distances[j, i] = min_distance
                # 更新最短距离的节点对索引
                closest_nodes = (component_nodes[i][idx1], component_nodes[j][idx2])
                distances[i, j] = distances[j, i] = dists.min()
    # 构建分量间的完全图
    complete_graph = nx.complete_graph(num_components)
    for i in range(num_components):
        for j in range(i + 1, num_components):
            complete_graph.add_edge(i, j, weight=distances[i, j])
    # 计算最小生成树
    mst = nx.minimum_spanning_tree(complete_graph, weight='weight')
    # 添加边到原图，以连接所有分量
    for edge in mst.edges():
        i, j = edge
        min_distance = np.inf
        best_pair = None
        # 在两个分量之间找到最近的节点对
        for node1 in component_nodes[i]:
            for node2 in component_nodes[j]:
                pos1 = np.array(graph.nodes[node1]['point'])
                pos2 = np.array(graph.nodes[node2]['point'])
                distance = np.linalg.norm(pos1 - pos2)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (node1, node2)
        graph.add_edge(*best_pair)
    return graph

def optimize_nodes_components(graph, min_size=20, image=None, show=False):
    batch_g = []
    kept_node_indices = []  # 记录每个图保留的节点索引
    for g in graph:
        g = connect_isolated_nodes(g)
        if show:
            print("2. Connect Isolated Nodes - 平均度数(双向):", sum(dict(g.degree()).values()) / g.number_of_nodes())
            visualize_graph_with_coordinates(g, title="2. Connect Isolated Nodes", image=image)
        g, kept = remove_small_components(g, min_size)
        if show:
            print("3. Remove Small Components - 平均度数(双向):", sum(dict(g.degree()).values()) / g.number_of_nodes())
            visualize_graph_with_coordinates(g, title="3. Remove Small Components", image=image)
        g = fast_connect_components(g)
        if show:
            print("4. Final Graph - 平均度数(双向):", sum(dict(g.degree()).values()) / g.number_of_nodes())
            visualize_graph_with_coordinates(g, title="4. Final Graph", image=image)
        batch_g.append(g)
        kept_node_indices.append(sorted(list(kept)))  # 转成 list 保存顺序
    return batch_g, kept_node_indices


# 对外API——动态阈值
def build_optimize_graph_with_cosine_similarity(keypoints, descriptors, scores, radius=20, percentile=50, min_size=10, device=torch.device('cpu'), image=None, show=False):
    '''
    percentile=0, min_size=1 时，就跟固定阈值一个效果。
    Args:
        keypoints:      关键点坐标
        descriptors:    关键点的特征描述符
        scores:         关键点的分数
        radius:         半径内的点可以生成边
        percentile:     特征描述符余弦相似度大于多少才连接边
        min_size:       孤立子图小于多少个节点就要被删除
        device:         GPU还是CPU上计算
    Returns: 处理后的子图
    '''
    graph = fast_build_graph_with_cosine_similarity(keypoints, descriptors, scores, radius, percentile)
    if show:
        print("1. Coarse Graph - 平均度数(双向):", sum(dict(graph[0].degree()).values()) / graph[0].number_of_nodes())
        visualize_graph_with_coordinates(graph[0], title="1. Coarse Graph", image=image)
    graph, kept_node_indices = optimize_nodes_components(graph, min_size=min_size, image=image, show=show)
    batch_graph = []
    for g in graph:
        points = torch.tensor(np.vstack([g.nodes[i]['point'] for i in g.nodes]), dtype=torch.float32, device=device)
        feats = torch.tensor(np.vstack([g.nodes[i]['feat'] for i in g.nodes]), dtype=torch.float32, device=device)
        scores = torch.tensor([g.nodes[i]['score'] for i in g.nodes], dtype=torch.float32, device=device)
        dgl_graph = dgl.from_networkx(g, device=device)
        dgl_graph.ndata['point'] = points
        dgl_graph.ndata['feat'] = feats
        dgl_graph.ndata['score'] = scores
        batch_graph.append(dgl_graph)
    return batch_graph, kept_node_indices

# 对外API——固定阈值
def build_graph_from_keypoints(keypoints, descriptors, scores, radius, device=torch.device('cpu')):
    '''固定阈值，改阈值内的顶点生成边'''
    graphs = create_graph_from_keypoints(keypoints, descriptors, scores, radius, device=torch.device('cpu'))
    return create_dgl_graph_from_batched_graph(graphs, device)


def build_graph_Delaunay(keypoints, descriptors, scores):
    keypoints = keypoints.cpu().numpy()
    descriptors = descriptors.permute(0, 2, 1).cpu().numpy()  # => (batch, number, dim)
    scores = scores.cpu().numpy()
    batch_graph = []
    for batched_idx in range(keypoints.shape[0]):
        kpts = keypoints[batched_idx]
        descs = descriptors[batched_idx]
        scrs = scores[batched_idx]
        graph = nx.Graph()
        for i in range(len(kpts)): graph.add_node(i, point=kpts[i], feat=descs[i], score=scrs[i])
        points = np.array([kp for kp in kpts])
        tri = Delaunay(points)
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    if not graph.has_edge(simplex[i], simplex[j]):
                        graph.add_edge(simplex[i], simplex[j])
        batch_graph.append(graph)
    return batch_graph

def build_graph_from_keypoints_Delaunay(keypoints, descriptors, scores, device=torch.device('cpu')):
    '''固定阈值，改阈值内的顶点生成边'''
    graph = build_graph_Delaunay(keypoints, descriptors, scores)
    batch_graph = []
    for g in graph:
        points = torch.tensor(np.vstack([g.nodes[i]['point'] for i in g.nodes]), dtype=torch.float32, device=device)
        feats = torch.tensor(np.vstack([g.nodes[i]['feat'] for i in g.nodes]), dtype=torch.float32, device=device)
        scores = torch.tensor([g.nodes[i]['score'] for i in g.nodes], dtype=torch.float32, device=device)
        dgl_graph = dgl.from_networkx(g, device=device)
        dgl_graph.ndata['point'] = points
        dgl_graph.ndata['feat'] = feats
        dgl_graph.ndata['score'] = scores
        batch_graph.append(dgl_graph)
    return batch_graph








