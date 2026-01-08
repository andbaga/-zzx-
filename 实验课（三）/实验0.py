import os
import json
import csv
import re
import time
import sys
import torch
import subprocess
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== 全局配置（方便用户修改）=====================
CONFIG = {
    # 核心模型路径 (请确认此路径正确)
    "model_name": r"C:\Users\21002\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B",
    "input_path": r"D:\张智炫的文档\数据挖掘与知识处理\实验课（三）\data\raw\Open-Patients.jsonl",
    
    # 输出目录
    "output_dir": "data/processed",
    "neo4j_dir": os.path.abspath("data/neo4j"), # 获取绝对路径
    
    # Neo4j 配置
    "neo4j_auth": "neo4j/password",
    "neo4j_ports": ("7474", "7687"),
    "docker_image": "neo4j:latest", # 使用官方镜像
    
    # 处理参数
    "max_text_length": 1024,
    "min_entity_types": 2,
    "test_mode": True,
    "test_limit": 4  # 测试数据量
}

# ===================== 数据处理工具类 =====================
class MedicalDataProcessor:
    @staticmethod
    def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
        data = []
        if not os.path.exists(file_path):
            print(f"错误：未找到原始数据文件 {file_path}")
            return data
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"成功加载 {len(data)} 条有效数据")
        return data

    @staticmethod
    def save_json_data(data: List[Dict[str, Any]], save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"JSON数据已保存至：{save_path}")

    @staticmethod
    def generate_neo4j_csv(data: List[Dict[str, Any]], output_dir: str) -> bool:
        """生成Neo4j兼容的节点/关系CSV文件"""
        os.makedirs(output_dir, exist_ok=True)
        node_path = os.path.join(output_dir, "nodes.csv")
        rel_path = os.path.join(output_dir, "relationships.csv")

        # 生成节点CSV（去重）
        seen_nodes = set()
        with open(node_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'type', 'name'])  # 表头
            
            entity_mapping = {
                "symptoms": "Symptom",
                "diseases": "Disease",
                "checks": "Check",
                "drugs": "Drug"
            }
            
            for article in data:
                for key, label in entity_mapping.items():
                    for item in article.get(key, []):
                        node_id = f"{key[:-1]}_{item}"  # 生成唯一ID（如symptom_咳嗽）
                        if node_id not in seen_nodes:
                            writer.writerow([node_id, label, item])
                            seen_nodes.add(node_id)

        # 生成关系CSV（去重）
        seen_rels = set()
        with open(rel_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['start_id', 'end_id', 'type'])  # 表头
            
            # 定义需要建立的实体关联
            rel_pairs = [
                ("symptoms", "diseases"),
                ("symptoms", "checks"),
                ("diseases", "checks"),
                ("diseases", "drugs")
            ]
            
            for article in data:
                for start_key, end_key in rel_pairs:
                    start_items = article.get(start_key, [])
                    end_items = article.get(end_key, [])
                    
                    for s_item in start_items:
                        for e_item in end_items:
                            start_id = f"{start_key[:-1]}_{s_item}"
                            end_id = f"{end_key[:-1]}_{e_item}"
                            rel_id = f"{start_id}-{end_id}"
                            
                            if rel_id not in seen_rels:
                                writer.writerow([start_id, end_id, "RELATED_TO"])
                                seen_rels.add(rel_id)

        print(f"Neo4j CSV文件已生成至：{output_dir}")
        return True

# ===================== 翻译工具 =====================
def translate_en_to_zh(text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device) -> str:
    """英文医学文本转中文（保留原翻译逻辑，优化提示词表述）"""
    if not text or not isinstance(text, str) or text.strip() == "":
        print("警告：无效的翻译输入文本")
        return text.strip() if text else ""
    
    # 优化后的翻译提示词（明确要求简洁准确）
    prompt = f"""请将以下英文医学文本精准翻译成中文，仅输出翻译结果，不添加任何解释、提问或额外内容：
英文原文：{text.strip()}
中文翻译："""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=CONFIG["max_text_length"]).to(device)
        
        generation_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        
        # 提取并清理翻译结果
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated = translated.replace(prompt, "").strip().split('\n')[0]
        return translated if translated else text.strip()
    
    except Exception as e:
        print(f"警告：翻译失败 - {str(e)}，返回原文")
        return text.strip()

# ===================== 医学实体提取器 =====================
class MedicalEntityExtractor:
    """整合模型加载与实体提取功能（优化错误处理）"""
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        self.device = self._get_device(device)
        self.tokenizer = self._load_tokenizer(model_name)
        self.model = self._load_model(model_name)

    def _get_device(self, device: Optional[torch.device]) -> torch.device:
        """自动选择计算设备（GPU优先）"""
        if device:
            return device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """加载分词器（处理pad_token）"""
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("分词器加载完成")
        return tokenizer

    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        """加载模型（优化内存配置）"""
        print(f"正在加载模型 {model_name}（设备：{self.device}）")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }
        
        # 根据设备设置精度
        if self.device.type == "cpu":
            model_kwargs["torch_dtype"] = torch.float32
        else:
            model_kwargs["torch_dtype"] = torch.float16

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            return model.to(self.device)
        except Exception as e:
            print(f"模型加载失败，尝试保守配置 - {str(e)}")
            model_kwargs["torch_dtype"] = torch.float32
            return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(self.device)

    def _clean_json_str(self, json_str: str) -> str:
        """清理JSON字符串（修复格式错误）"""
        # 替换单引号、去除尾部逗号、修复键名引号
        json_str = json_str.replace("'", '"').replace(",]", "]").replace(",}", "}")
        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)
        # 提取第一个完整JSON对象
        json_matches = re.findall(r'\{[^{}]*\}', json_str, re.DOTALL)
        if json_matches:
            json_str = json_matches[0]
        return json_str.strip()

    def _extract_manual(self, text: str) -> Optional[Dict[str, List[str]]]:
        """JSON解析失败时手动提取实体（优化正则匹配）"""
        result = {"symptoms": [], "diseases": [], "checks": [], "drugs": []}
        entity_types = list(result.keys())
        
        for et in entity_types:
            # 多模式匹配实体列表
            patterns = [
                rf'"{et}"\s*:\s*\[(.*?)\]',
                rf'{et}\s*:\s*\[(.*?)\]',
                rf'{et}\s*=\s*\[(.*?)\]'
            ]
            
            for pat in patterns:
                matches = re.findall(pat, text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    if not match.strip():
                        continue
                    # 提取引号内或逗号分隔的实体
                    items = re.findall(r'"([^"]*)"', match) or re.findall(r"'([^']*)'", match)
                    if not items:
                        items = [i.strip() for i in match.split(',') if i.strip()]
                    result[et].extend([i for i in items if i])
        
        # 去重并过滤空值
        for et in entity_types:
            result[et] = list(set([i for i in result[et] if i]))
        
        # 检查是否满足最小实体类型数
        valid_types = sum(1 for v in result.values() if v)
        return result if valid_types >= CONFIG["min_entity_types"] else None

    def extract(self, text: str) -> Optional[Dict[str, List[str]]]:
        """核心实体提取方法（整合JSON解析与手动提取）"""
        # 文本截断
        if len(text) > CONFIG["max_text_length"]:
            text = text[:CONFIG["max_text_length"]]
            print(f"提示：文本过长，已截断至 {CONFIG['max_text_length']} 字符")
        
        # 优化后的实体提取提示词
        prompt = f"""请从以下医学病例中提取4类实体：症状（symptoms）、疾病（diseases）、检查（checks）、药物（drugs）。
要求：
1. 无对应实体则返回空数组
2. 仅输出JSON格式，无任何额外内容
3. JSON键名必须为指定英文（symptoms/diseases/checks/drugs）
4. 数组项用双引号包裹，逗号分隔

病例文本：{text}

JSON输出："""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=CONFIG["max_text_length"]).to(self.device)
            generation_kwargs = {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.1,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取JSON部分
            json_str = self._clean_json_str(response)
            result = json.loads(json_str)
            
            # 验证格式并去重
            for et in ["symptoms", "diseases", "checks", "drugs"]:
                if et not in result:
                    result[et] = []
                result[et] = list(set([i.strip() for i in result[et] if i.strip()]))
            
            # 检查最小实体类型数
            valid_types = sum(1 for v in result.values() if v)
            if valid_types >= CONFIG["min_entity_types"]:
                return result
            else:
                print(f"提示：仅提取到 {valid_types} 类实体，低于最小要求")
                return None
        
        except json.JSONDecodeError:
            print("警告：JSON解析失败，尝试手动提取")
            return self._extract_manual(response)
        except Exception as e:
            print(f"错误：实体提取失败 - {str(e)}")
            return None

# ===================== Neo4j自动化工具 =====================
class Neo4jAutoDeploy:
    """整合Neo4j容器启动、文件拷贝、数据导入（含智能重试机制）"""
    
    @staticmethod
    def run_docker_cmd(cmd: List[str], check_error=True, suppress_output=False) -> bool:
        """执行Docker命令"""
        try:
            if not suppress_output:
                print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=check_error, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            if check_error and not suppress_output:
                print(f"❌ 命令执行失败: {e.stderr.strip()}")
            return False

    def start_container(self):
        """启动Neo4j容器"""
        image_name = "neo4j:latest" 
        
        # 1. 强制清理旧容器 (确保环境干净)
        print("清理旧容器环境...")
        self.run_docker_cmd(["docker", "stop", "neo4j"], check_error=False, suppress_output=True)
        self.run_docker_cmd(["docker", "rm", "neo4j"], check_error=False, suppress_output=True)
        
        # 2. 启动新容器
        # 注意：这里我们只挂载 data 目录做持久化，import 目录我们后面用 docker cp 模拟手动操作
        # 这样可以避免 Windows 挂载导致的文件读取权限问题
        abs_neo4j_dir = os.path.abspath(CONFIG['neo4j_dir'])
        ports = CONFIG["neo4j_ports"]
        
        run_cmd = [
            "docker", "run", "--name", "neo4j",
            "-p", f"{ports[0]}:{ports[0]}", "-p", f"{ports[1]}:{ports[1]}",
            "-v", f"{abs_neo4j_dir}:/data", 
            "-d", "-e", f"NEO4J_AUTH={CONFIG['neo4j_auth']}",
            image_name
        ]
        
        print(f"🚀 正在启动容器 {image_name}...")
        if self.run_docker_cmd(run_cmd):
            print("✅ 容器启动指令已发送")
            return True
        return False

    def wait_for_neo4j_and_import(self, csv_dir: str):
        """核心逻辑：等待数据库就绪 -> 拷贝文件 -> 导入数据"""
        
        # 1. 模拟 docker cp 操作
        print("\n[自动化] 正在将CSV文件拷贝至容器内部...")
        for csv_file in ["nodes.csv", "relationships.csv"]:
            src_path = os.path.join(csv_dir, csv_file)
            if not os.path.exists(src_path):
                print(f"❌ 错误：找不到文件 {src_path}")
                return False
            
            # 使用 docker cp 命令
            cp_cmd = ["docker", "cp", src_path, "neo4j:/var/lib/neo4j/import/"]
            if not self.run_docker_cmd(cp_cmd):
                print(f"❌ 文件 {csv_file} 拷贝失败")
                return False
        print("✅ 文件拷贝完成")

        # 2. 准备导入命令
        # 你的手动命令非常完美，我们直接复用它
        cypher_query = (
            "LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row "
            "CREATE (n:MedicalEntity {id: row.id, type: row.type, name: row.name}); "
            "LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row "
            "MATCH (start:MedicalEntity {id: row.start_id}) "
            "MATCH (end:MedicalEntity {id: row.end_id}) "
            "CREATE (start)-[r:RELATED_TO]->(end);"
        )
        
        import_cmd = [
            "docker", "exec", "neo4j", "cypher-shell",
            "-u", CONFIG["neo4j_auth"].split('/')[0],
            "-p", CONFIG["neo4j_auth"].split('/')[1],
            cypher_query
        ]

        # 3. 循环重试机制 (专门解决 Connection refused)
        print("\n[自动化] 开始尝试连接数据库并导入 (最多等待 5 分钟)...")
        max_retries = 30
        for i in range(1, max_retries + 1):
            sys.stdout.write(f"\r⏳ 第 {i}/{max_retries} 次尝试连接 Neo4j... ")
            sys.stdout.flush()
            
            try:
                # 尝试执行导入
                result = subprocess.run(import_cmd, check=True, capture_output=True, text=True)
                print("\n\n🎉 导入成功！(Exit Code: 0)")
                return True
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.lower()
                # 如果是连接错误，说明还在启动中，等待并重试
                if "connection refused" in error_msg or "failed to connect" in error_msg or "connect to localhost" in error_msg:
                    time.sleep(10) # 等待10秒再试
                else:
                    # 如果是其他错误（比如语法错误），直接报错停止
                    print(f"\n❌ 发生非连接错误，停止重试:\n{e.stderr}")
                    return False
        
        print("\n❌ 超时：Neo4j 启动时间过长，请检查 Docker 日志。")
        return False

# ===================== 主流程 =====================
def main():
    print("="*60)
    print("          医学实体知识图谱构建流程启动          ")
    print("="*60)

    # 1. 初始化核心组件
    print("\n【步骤1/5】初始化模型与工具...")
    extractor = MedicalEntityExtractor(CONFIG["model_name"])
    data_processor = MedicalDataProcessor()
    neo4j_deployer = Neo4jAutoDeploy()

    # 2. 加载原始数据
    print("\n【步骤2/5】加载原始数据...")
    raw_data = data_processor.load_jsonl_data(CONFIG["input_path"])
    if not raw_data:
        print("错误：无有效原始数据，流程终止")
        return

    # 3. 处理数据（翻译+实体提取）
    print("\n【步骤3/5】处理数据（翻译+实体提取）...")
    processed_data = []
    # 逻辑：如果是测试模式，只取前N条；否则全量
    limit = CONFIG["test_limit"] if CONFIG["test_mode"] else len(raw_data)
    target_data = raw_data[:limit]
    
    for i, item in enumerate(target_data, 1):
        print(f"\n--- 处理进度 {i}/{len(target_data)} ---")
        
        original_text = item.get("description", "")
        if not original_text:
            continue
        
        # 翻译
        translated = translate_en_to_zh(original_text, extractor.model, extractor.tokenizer, extractor.device)
        
        # 提取实体
        entities = extractor.extract(translated)
        if entities:
            print(f"  成功提取实体: {sum(len(v) for v in entities.values())} 个")
            processed_item = {
                "id": item.get("_id", f"item_{i}"),
                "original": original_text,
                "translated": translated,
                **entities
            }
            processed_data.append(processed_item)
        else:
            print("  未提取到有效实体")

    if not processed_data:
        print("❌ 错误：无有效处理结果，流程终止")
        return

    # 4. 保存结果
    print("\n【步骤4/5】保存处理结果...")
    json_path = os.path.join(CONFIG["output_dir"], "processed_articles.json")
    data_processor.save_json_data(processed_data, json_path)
    
    neo4j_csv_dir = os.path.join(CONFIG["output_dir"], "neo4j")
    data_processor.generate_neo4j_csv(processed_data, neo4j_csv_dir)

    # 5. Neo4j 全自动部署
    print("\n【步骤5/5】Neo4j 自动化部署与导入...")
    
    # 第一步：启动容器
    if neo4j_deployer.start_container():
        # 第二步：智能等待并导入（整合了cp和loop逻辑）
        if neo4j_deployer.wait_for_neo4j_and_import(neo4j_csv_dir):
            print("\n" + "="*60)
            print("🎉 恭喜！全流程执行成功！")
            print(f"👉 知识图谱查看地址: http://localhost:{CONFIG['neo4j_ports'][0]}")
            print(f"👉 登录账号: {CONFIG['neo4j_auth'].split('/')[0]}")
            print(f"👉 登录密码: {CONFIG['neo4j_auth'].split('/')[1]}")
            print("="*60)
        else:
            print("❌ 导入阶段失败")
    else:
        print("❌ 容器启动失败")

if __name__ == "__main__":
    main()