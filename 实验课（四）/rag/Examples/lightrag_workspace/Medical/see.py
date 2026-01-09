import networkx as nx
import matplotlib.pyplot as plt

# 1. 加载 GraphML 文件
G = nx.read_graphml(r"D:\张智炫的文档\数据挖掘与知识处理\实验课（四）\rag\Examples\lightrag_workspace\Medical\graph_chunk_entity_relation.graphml")

# 打印基本信息
print(f"总节点数: {G.number_of_nodes()}")
print(f"总边数: {G.number_of_edges()}")

# 2. 筛选核心节点 (例如 Top 50)
# 根据度数对节点进行排序
degrees = dict(G.degree())
top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:50]

# 构建子图
subgraph = G.subgraph(top_nodes)

# 3. 绘图
plt.figure(figsize=(12, 12)) # 设置画布大小

# 使用弹簧布局 (k值调整节点间距)
pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)

# 根据度数设置节点大小
node_sizes = [subgraph.degree(n) * 100 for n in subgraph.nodes()]

# 绘制节点、边和标签
nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color='skyblue', alpha=0.9)
nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.5, edge_color='gray')
nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold')

plt.title("Knowledge Graph Visualization (Top 50 Nodes)")
plt.axis('off') # 关闭坐标轴
plt.show()