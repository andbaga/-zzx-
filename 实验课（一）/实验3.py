from gensim.models import Word2Vec
import numpy as np

# 1. word2vec相似度计算
w2v_model = Word2Vec.load('word2vec_amazon.model')
word1, word2 = 'great', 'excellent'
if word1 in w2v_model.wv and word2 in w2v_model.wv:
    w2v_similarity = w2v_model.wv.similarity(word1, word2)
    print(f"word2vec - '{word1}'与'{word2}'的相似度：{w2v_similarity:.4f}")

# 2. node2vec相似度计算（解析.emb文件，跳过异常行）
def load_emb_file(emb_path, encoding='utf-8'):
    """手动加载.emb文件，处理格式异常"""
    node_vectors = {}
    with open(emb_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        # 解析文件头（第一行：节点数 向量维度）
        header = lines[0].strip().split()
        if len(header) != 2 or not header[0].isdigit() or not header[1].isdigit():
            raise ValueError("emb文件头格式错误，第一行必须是'节点数 向量维度'")
        vocab_size, vec_dim = int(header[0]), int(header[1])
        
        # 解析每个节点的向量
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 验证：节点名 + 向量值（向量值数量 = vec_dim）
            if len(parts) != vec_dim + 1:
                continue
            node = parts[0]
            try:
                # 向量部分必须全部转换为浮点数
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                node_vectors[node] = vec
            except ValueError:
                print(f"跳过非数字向量行：{line}")
                continue
    return node_vectors, vec_dim

# 加载修复后的emb文件
try:
    n2v_vectors, vec_dim = load_emb_file('node2vec_linkedin.emb')
    print(f"\nnode2vec加载成功：有效节点数={len(n2v_vectors)}，向量维度={vec_dim}")
    
    # 计算节点相似度（选前2个有效节点）
    valid_nodes = list(n2v_vectors.keys())[:2]
    if len(valid_nodes) >= 2:
        node1, node2 = valid_nodes
        vec1, vec2 = n2v_vectors[node1], n2v_vectors[node2]
        # 自定义余弦相似度计算
        def cosine_similarity(vec1, vec2):
            dot = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
        n2v_similarity = cosine_similarity(vec1, vec2)
        print(f"node2vec - '{node1}'与'{node2}'的相似度：{n2v_similarity:.4f}")
    else:
        print("有效节点数不足，无法计算相似度")
except Exception as e:
    print(f"node2vec加载失败：{str(e)}")
