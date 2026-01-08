import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import re

# 1. 数据加载与混合数据清洗（不跳过有效行，仅清理数值干扰）
posting_df = pd.read_csv(
    r"D:/张智炫的文档/数据挖掘与知识处理/实验课（一）/LinkedIn Job Postings/postings.csv",
    encoding='latin-1',
    on_bad_lines='warn'  # 在遇到格式不正确或无法解析的行时发出警告信息
)

# 定义混合数据清洗函数：过滤纯数值/含大量数值的无效技能，保留文本
def clean_mixed_data(text):
    if pd.isna(text):  # 处理空值
        return []
    # 分割技能（处理逗号分隔，兼容可能的特殊分隔符）
    skills = str(text).split(',')
    cleaned_skills = []
    for skill in skills:
        skill = skill.strip()
        # 过滤条件：1. 非空 2. 长度>2 3. 不是纯数值 4. 数值字符占比<50%（避免文本+数值混合）
        if skill and len(skill) > 2:
            # 统计数值字符占比
            num_chars = len(re.findall(r'[-+]?\d+\.?\d*', skill))
            if not skill.replace('.', '').replace('-', '').isdigit() and (num_chars / len(skill)) < 0.5:
                cleaned_skills.append(skill)
    return cleaned_skills

# 构造职业-技能边列表
job_skills = []
invalid_rows = 0  # 统计完全无效的行
for idx, row in posting_df.iterrows():
    # 处理职位名称（空值用job_索引填充，保证节点唯一性）
    job_title = row['title'] if pd.notna(row['title']) else f'job_{idx}'
    # 清洗skills_desc：剔除数值混合的无效技能，保留纯文本技能
    skills = clean_mixed_data(row['skills_desc'])
    if not skills:
        invalid_rows += 1
        continue
    # 构建有效边
    for skill in skills:
        job_skills.append((job_title, skill))

# 去重重复边（同一职位-技能对仅保留1条）
job_skills = list(set(job_skills))
print(f"去重后职业-技能边总数：{len(job_skills)}")
print(f"完全无效的行（无有效技能）：{invalid_rows}")

# 构建无向图（添加节点过滤，避免孤立节点）
G = nx.Graph()
G.add_edges_from(job_skills)
# 过滤孤立节点（仅保留有边连接的节点，优化图结构）
G = G.subgraph([node for node in G.nodes() if G.degree(node) > 0])
print(f"过滤孤立节点后总节点数：{len(G.nodes())}")
print(f"过滤孤立节点后总边数：{len(G.edges())}")

# 2. 优化后的node2vec模型训练（适配多核CPU，平衡效率与效果）
if len(G.edges()) > 0:
    node2vec = Node2Vec(
        G,
        dimensions=100,        # 核心特征维度
        walk_length=30,        # 随机游走长度
        num_walks=200,         # 每个节点游走次数
        p=1, q=1,              # 保持原游走策略
        workers=10             # 并行计算线程数
    )

    # Word2Vec训练参数优化
    n2v_model = node2vec.fit(
        window=5,             # 上下文窗口大小
        min_count=2,          # 过滤低频节点（出现<2次的节点）
        batch_words=1000,     # 批处理量
        sg=1,                 # 固定Skip-gram模型（node2vec默认）
        epochs=10             # 训练轮数
    )

    # 保存节点向量（保持原输出路径）
    n2v_model.wv.save_word2vec_format('node2vec_linkedin.emb')
    print("节点向量已保存至 node2vec_linkedin.emb")

    # 3. 验证：获取多个示例节点的向量（确保功能正常）
    sample_nodes = list(G.nodes())[:2]  # 取前2个节点验证
    for node in sample_nodes:
        node_vector = n2v_model.wv[node]
        print(f"\n示例节点 '{node}' 的向量信息：")
        print(f"向量维度：{len(node_vector)}")
        print(f"向量前10维：{np.round(node_vector[:10], 4)}")
else:
    print("警告：图中无有效边，请检查数据集或清洗逻辑！")