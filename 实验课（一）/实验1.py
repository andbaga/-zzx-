import pandas as pd
import re
from nltk.corpus import stopwords  # 从NLTK库的语料库模块中导入停用词（在文本分析中通常被过滤掉的常见词汇）列表
from gensim.models import Word2Vec
import nltk

# 下载必要资源
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# 1. 数据加载与预处理
df1 = pd.read_csv(r"D:/张智炫的文档/数据挖掘与知识处理/实验课（一）/Amazon reviews/test.csv", header=None, names=['rating', 'title', 'review'])
df2 = pd.read_csv(r"D:/张智炫的文档/数据挖掘与知识处理/实验课（一）/Amazon reviews/train_part_1.csv", header=None, names=['rating', 'title', 'review'])
df3 = pd.read_csv(r"D:/张智炫的文档/数据挖掘与知识处理/实验课（一）/Amazon reviews/train_part_2.csv", header=None, names=['rating', 'title', 'review'])
reviews_df = pd.concat([df1, df2, df3], ignore_index=True)
reviews_df = reviews_df.dropna(subset=['review'])
# 去重重复评论
reviews_df = reviews_df.drop_duplicates(subset=['review'])

# 文本清洗
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 去除非字母字符
    tokens = text.split()  # 空格分词
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

# 预处理和corpus生成
reviews_df['processed_text'] = reviews_df['review'].apply(preprocess_text)
corpus = reviews_df['processed_text'].tolist()  # 将processed_text列转换为列表形式
corpus = [tokens for tokens in corpus if len(tokens) > 0]
print(f"有效语料条数：{len(corpus)}")

# 2. 构建word2vec模型
w2v_model = Word2Vec(
    sentences=corpus,
    vector_size=100,  # 词向量维度
    window=5,         # 上下文窗口大小
    min_count=5,      # 最小词频（过滤低频词）
    sg=1,             # 1=使用Skip-gram，0=CBOW
    workers=12,       # 并行计算线程数
    epochs=3,         # 训练轮数
    seed=42           # 随机种子
)
w2v_model.save('word2vec_amazon.model')

# 3. 示例：获取词向量与相似词
try:
    print("'great'的词向量前10维：", w2v_model.wv['great'][:10])
    print("与'great'最相似的5个词：", w2v_model.wv.most_similar('great', topn=5))
except KeyError:
    # 避免“great”被过滤导致报错
    common_word = next(iter(w2v_model.wv.index_to_key))
    print(f"\n高频词'{common_word}'的词向量前10维：", w2v_model.wv[common_word][:10])
    print(f"与'{common_word}'最相似的5个词：", w2v_model.wv.most_similar(common_word, topn=5))