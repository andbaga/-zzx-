import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
import random

# ==========================================
# 全局绘图风格设置 (让图表更好看)
# ==========================================
# 设置为包括网格的白底风格，适合学术展示
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
# 解决中文显示问题（如果你的环境支持中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_scatter(df, title, hue_col=None):
    """
    通用美化绘图函数
    """
    plt.figure(figsize=(12, 8), dpi=120)  # 高分辨率
    
    # 使用Seaborn绘制散点图
    if hue_col:
        # 如果有分类，使用特定色板
        scatter = sns.scatterplot(
            data=df, x='x', y='y', 
            hue=hue_col, 
            palette='viridis', # 现代配色
            s=80,              # 点的大小
            alpha=0.8,         # 透明度
            edgecolor='w'      # 白色描边，增加对比度
        )
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Cluster/Category")
    else:
        # 纯色模式
        scatter = sns.scatterplot(
            data=df, x='x', y='y', 
            color='#3498db', s=60, alpha=0.6
        )

    # 标注部分文本（避免所有文本都标注重叠）
    # 策略：随机选取部分或者按重要性选取
    texts = []
    step = max(1, len(df) // 40) # 动态控制标签密度
    for i in range(0, len(df), step):
        texts.append(plt.text(df.iloc[i]['x']+0.2, df.iloc[i]['y'], df.iloc[i]['word'], fontsize=9))

    plt.title(title, fontsize=16, weight='bold', pad=20)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.show()

# ==========================================
# 任务 1: Word2Vec 语义聚类可视化 (基于实验1)
# ==========================================
def visualize_word2vec_clusters(model_path):
    print(f"--- 正在加载 Word2Vec 模型: {model_path} ---")
    try:
        model = Word2Vec.load(model_path)
    except FileNotFoundError:
        print(f"错误：未找到 {model_path}，请先运行实验1。")
        return

    # 技巧：为了让可视化好看且有意义，我们不随机画点
    # 而是选取几个具有代表性的"中心词"，画出它们周围的词，看是否能明显聚类
    target_words = ['good', 'bad', 'price', 'battery', 'service'] # 你可以根据你的数据修改这些词
    
    words_to_plot = []
    labels = []
    vectors = []

    print("正在提取语义群落...")
    for target in target_words:
        if target in model.wv:
            # 添加中心词
            words_to_plot.append(target)
            labels.append(target) # 标签就是它自己，作为一类
            vectors.append(model.wv[target])
            
            # 添加最相似的Top 15词
            sim_words = model.wv.most_similar(target, topn=15)
            for word, _ in sim_words:
                words_to_plot.append(word)
                labels.append(target) # 标记这个词属于 target 这个阵营
                vectors.append(model.wv[word])
        else:
            print(f"警告：词汇 '{target}' 不在模型词典中，跳过。")

    if not vectors:
        print("没有提取到有效向量，请检查模型或关键词。")
        return

    # t-SNE 降维
    print("正在进行 t-SNE 降维 (这可能需要几秒钟)...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(vectors)-1), random_state=42, init='pca', learning_rate='auto')
    reduced_vecs = tsne.fit_transform(np.array(vectors))

    # 构建绘图数据
    df_plot = pd.DataFrame(reduced_vecs, columns=['x', 'y'])
    df_plot['word'] = words_to_plot
    df_plot['cluster'] = labels

    print("正在绘制 Word2Vec 语义分布图...")
    plot_scatter(df_plot, "Experiment 1 Result: Word2Vec Semantic Clusters", hue_col='cluster')


# ==========================================
# 任务 2: Node2Vec 网络结构可视化 (基于实验2)
# ==========================================
def load_emb_file_custom(emb_path):
    """
    加载emb文件，增加了维度校验，防止因数据格式错误导致的报错
    """
    vectors = []
    names = []
    expected_dim = None # 预期的向量维度
    
    with open(emb_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 1. 尝试解析第一行 header 获取维度
        # 通常第一行格式为: number_of_nodes dimensions
        if len(lines) > 0:
            header = lines[0].strip().split()
            if len(header) == 2 and header[0].isdigit() and header[1].isdigit():
                expected_dim = int(header[1])
                print(f"检测到模型维度为: {expected_dim}")
                # 去掉第一行header
                lines = lines[1:]
            else:
                # 如果第一行不是header，可能是无header格式，此时不做操作，稍后自动推断
                pass

        # 过滤空行
        data_lines = [line.strip() for line in lines if line.strip()]
        
        # 2. 采样逻辑
        sample_size = 600
        if len(data_lines) > sample_size:
            print(f"检测到节点数量庞大 ({len(data_lines)})，随机采样 {sample_size} 个节点进行可视化...")
            data_lines = random.sample(data_lines, sample_size)
        
        # 3. 解析数据
        for line in data_lines:
            parts = line.split()
            # 至少要有 节点名 + 1个向量值
            if len(parts) < 2: 
                continue
                
            node_name = parts[0]
            try:
                # 尝试将剩余部分转为浮点数
                current_vec = [float(x) for x in parts[1:]]
                
                # 如果还没有确定维度，以第一条成功解析的数据为准
                if expected_dim is None:
                    expected_dim = len(current_vec)
                
                # 核心修复：检查向量长度是否等于预期维度
                if len(current_vec) != expected_dim:
                    # 长度不对（可能是名字里有空格导致分割多了，或者行数据缺失），跳过该行
                    continue
                
                vec = np.array(current_vec, dtype=np.float32)
                names.append(node_name)
                vectors.append(vec)
            except ValueError:
                # 转换浮点数失败，跳过
                continue
                
    print(f"成功加载有效节点向量: {len(vectors)} 个")
    return np.array(vectors), names

def visualize_node2vec_structure(emb_path):
    print(f"\n--- 正在加载 Node2Vec 嵌入: {emb_path} ---")
    try:
        vectors, names = load_emb_file_custom(emb_path)
    except FileNotFoundError:
        print(f"错误：未找到 {emb_path}，请先运行实验2。")
        return

    if len(vectors) == 0:
        print("未加载到有效节点向量。")
        return

    # t-SNE 降维
    print("正在进行 t-SNE 降维...")
    # perplexity 参数需要小于样本数
    perp = min(30, len(vectors) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    reduced_vecs = tsne.fit_transform(vectors)

    # 尝试区分节点类型（基于实验2数据，如果是 'Full-time' 这种是大写开头单词可能是属性，纯文本可能是技能）
    # 这是一个简单的启发式规则，用来给图上色，让它好看一点
    # 假设：包含数字的或者很短的可能是ID，纯字母的是技能
    node_types = []
    for name in names:
        if name.isdigit():
            node_types.append('ID/Job')
        elif any(char.isdigit() for char in name):
            node_types.append('Mixed')
        else:
            node_types.append('Skill/Text')

    df_plot = pd.DataFrame(reduced_vecs, columns=['x', 'y'])
    df_plot['word'] = names
    df_plot['type'] = node_types

    print("正在绘制 Node2Vec 结构分布图...")
    plot_scatter(df_plot, "Experiment 2 Result: Node2Vec Graph Embeddings (Sampled)", hue_col='type')


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    print("=== 开始执行实验 4：模型可视化分析 ===\n")
    
    # 1. 可视化 Amazon 评论词向量
    # 假设实验1保存的模型名为 'word2vec_amazon.model'
    visualize_word2vec_clusters(r"C:/Users/21002/Desktop/PythonTest/word2vec_amazon.model")

    # 2. 可视化 LinkedIn 职位图向量
    # 假设实验2保存的文件名为 'node2vec_linkedin.emb'
    visualize_node2vec_structure(r"C:/Users/21002/Desktop/PythonTest/node2vec_linkedin.emb")
    
    print("\n=== 实验 4 完成 ===")