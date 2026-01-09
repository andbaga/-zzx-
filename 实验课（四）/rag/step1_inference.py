import pandas as pd
import requests
import json
import os
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（四）\rag\Datasets\Questions\medical_questions.json"
OUTPUT_FILE = r"D:\张智炫的文档\数据挖掘与知识处理\实验课（四）\rag\Evaluation\eval_ready_data.json"
SERVER_URL = "http://localhost:8000/query"
# ===========================================

def load_data(filepath):
    if filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    elif filepath.endswith('.json'):
        try:
            return pd.read_json(filepath)
        except ValueError:
            return pd.read_json(filepath, lines=True)
    else:
        raise ValueError("不支持的文件格式")

def run_inference():
    print(f"正在读取数据: {INPUT_FILE}")
    try:
        df = load_data(INPUT_FILE)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # >>>>>>>>>> 关键修改在这里：只取前15条数据 <<<<<<<<<<
    df = df.head(15) 
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    print(f"包含列名: {df.columns.tolist()}")

    col_map = {}
    for col in df.columns:
        if col.lower() in ['question', 'query', 'input']:
            col_map['question'] = col
        if col.lower() in ['answer', 'output', 'ground_truth', 'ground_truth_answer']:
            col_map['ground_truth'] = col
    
    if 'question' not in col_map:
        print("错误：无法在数据中找到 'question' 列")
        return

    results = []
    print(f"开始推理，共 {len(df)} 条数据 (已截断)...")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row[col_map['question']]
        ground_truth = row[col_map['ground_truth']] if 'ground_truth' in col_map else ""

        try:
            response = requests.post(
                SERVER_URL, 
                json={"query": question, "mode": "hybrid"},
                timeout=300
            )
            
            if response.status_code == 200:
                resp_json = response.json()
                
                result_item = {
                    "id": str(index),
                    "question_type": "Medical",
                    "question": question,
                    "ground_truth": ground_truth,
                    "evidence": [ground_truth],
                    "generated_answer": resp_json.get("answer", ""),
                    "context": resp_json.get("context", []) 
                }
                results.append(result_item)
            else:
                print(f"Server Error {response.status_code}")
                
        except Exception as e:
            print(f"Request Error: {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 推理完成！生成的数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_inference()