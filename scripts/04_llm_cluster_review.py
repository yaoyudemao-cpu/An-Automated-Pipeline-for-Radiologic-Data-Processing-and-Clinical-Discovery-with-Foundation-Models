import pandas as pd
import requests
import json
import os
import time
from tqdm import tqdm
from pipeline_config import (
    STEP3_CONCLUSION_TEMPLATE_CSV,
    STEP3_IMPRESSION_TEMPLATE_CSV,
    STEP4_CONCLUSION_REVIEW_CSV,
    STEP4_IMPRESSION_REVIEW_CSV,
)
from pipeline_utils import ensure_directory

# 配置Ollama API地址和模型名称
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "gpt-oss:120b"          # 请根据实际部署的模型名称修改

def call_llm(prompt, temperature=0.0, max_retries=3):
    """
    调用本地Ollama模型，返回响应文本
    """
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "top_p": 0.5
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=data, timeout=600)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"API请求失败 (状态码 {response.status_code})，尝试 {attempt+1}/{max_retries}")
                time.sleep(2)
        except Exception as e:
            print(f"请求异常: {e}，尝试 {attempt+1}/{max_retries}")
            time.sleep(2)
    return ""

def parse_diagnosis_list(diagnosis_str):
    """
    解析聚类内诊断结论字符串，格式如 "椎体骨质增生(12); 骨质增生(8); 椎体边缘增生(5)"
    返回一个列表，每个元素为 "诊断名称 (频率)" 的字符串
    """
    if pd.isna(diagnosis_str) or not diagnosis_str:
        return []
    parts = diagnosis_str.split(';')
    lines = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 提取诊断名称和频率，假设格式为 "诊断名称(频率)"
        if '(' in part and part.endswith(')'):
            name = part[:part.rfind('(')].strip()
            freq = part[part.rfind('(')+1:-1].strip()
        else:
            name = part
            freq = "1"
        lines.append(f"{name} ({freq})")
    return lines

def build_prompt(cluster_id, representative, diagnosis_lines):
    """
    构建发送给LLM的Prompt
    """
    diagnosis_list_text = "\n".join(diagnosis_lines)
    prompt = f"""
你是一个医疗术语规范化专家。你的任务是对一组含义相近的诊断结论进行审核，并给出一个统一的规范名称，同时判断该聚类应该“接受”、“修改”还是“拆分”。

输入是一个聚类内的所有诊断结论及其出现频率（括号内为频率）。请根据临床知识，将它们合并成一个最合适、最规范的诊断名称。

这些诊断结论，都是来源于脊柱的影像报告（x光、CT、MRI），请注意修正时名称的范围，请不要过度修改以免造成错误。
要求：
- 规范名称应简洁、准确，符合医学书写习惯，其中一些缩写，尤其是英文专业词汇不需要展开。
- 如果聚类内的诊断结论存在明显差异无法合并，请选择“拆分”，并给出拆分后建议的多个规范名称，为了后续处理所有“拆分”必须按照以下形式，“拆分后规范名称：原诊断结论1，诊断结论2...”，每个拆分后的规范名称用分号分隔。
- 如果聚类内的诊断结论可以接受当前的代表名称，请选择“接受”，规范名称可直接使用代表名称。
- 如果代表名称不够规范，需要修改，请选择“修改”，并提供新的规范名称。
- 输出格式必须为JSON，包含以下字段：
  {{
    "final_name": "规范名称（如果拆分为多个，请用分号分隔）",
    "review_result": "接受/修改/拆分",
    "remarks": "简要说明理由或特殊说明"
  }}
  只输出JSON，不要添加任何其他文字。

以下是聚类信息：
聚类ID：{cluster_id}
代表诊断：{representative}
聚类内诊断结论及频率：
{diagnosis_list_text}

请输出JSON。
"""
    return prompt.strip()

def build_retry_prompt(original_prompt, error_hint=""):
    """
    构建重试时的提示，要求模型严格输出JSON
    """
    retry_prompt = f"""
你之前的返回无法解析为有效的JSON。请重新输出，必须严格遵守以下要求：
- 只输出一个JSON对象，不要包含任何其他文字、解释或Markdown代码块标记。
- JSON对象必须包含字段：final_name, review_result, remarks。
- final_name: 字符串，规范名称（如果拆分为多个，用分号分隔）。
- review_result: 字符串，只能是“接受”、“修改”或“拆分”。
- remarks: 字符串，简要说明理由。

原始任务如下，请再次处理并仅输出JSON：
{original_prompt}
"""
    return retry_prompt.strip()

def extract_json_from_response(response_text):
    """
    从模型返回的文本中提取JSON部分（处理可能的Markdown代码块）
    """
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            try:
                return json.loads(json_str)
            except:
                pass
    return None

def llm_audit_clusters(input_csv_path, output_csv_path, save_interval=5, max_parse_attempts=3):
    """
    处理单个CSV文件的LLM审核
    """
    print(f"\n正在处理文件: {os.path.basename(input_csv_path)}")
    print(f"读取CSV文件: {input_csv_path}")
    df = pd.read_csv(input_csv_path, encoding='utf-8-sig')
    
    required_cols = ['聚类ID', '聚类代表', '聚类内诊断结论']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件中缺少必要的列: {col}")
    
    if os.path.exists(output_csv_path):
        print(f"警告: 输出文件 {output_csv_path} 已存在，将被覆盖。")
    
    # 初始化结果列
    if '最终规范名称' not in df.columns:
        df['最终规范名称'] = ""
    if '审核结果' not in df.columns:
        df['审核结果'] = ""
    if '备注' not in df.columns:
        df['备注'] = ""
    if '审核状态' not in df.columns:
        df['审核状态'] = "待处理"
    
    total_rows = len(df)
    print(f"开始处理 {total_rows} 个聚类...")
    
    for idx in tqdm(range(total_rows), desc=f"处理进度", unit="聚类"):
        row = df.iloc[idx]
        cluster_id = row['聚类ID']
        representative = row['聚类代表']
        diagnosis_str = row['聚类内诊断结论']
        
        if df.at[idx, '审核状态'] in ["成功", "重试成功"]:
            continue
        
        diagnosis_lines = parse_diagnosis_list(diagnosis_str)
        if not diagnosis_lines:
            print(f"聚类 {cluster_id} 的诊断列表为空，跳过")
            df.at[idx, '审核状态'] = "诊断列表为空"
            continue
        
        original_prompt = build_prompt(cluster_id, representative, diagnosis_lines)
        success = False
        final_result = None
        status = "解析失败"
        
        for attempt in range(max_parse_attempts):
            if attempt == 0:
                temperature = 0.0
                current_prompt = original_prompt
            else:
                temperature = min(0.1 * attempt, 0.5)
                current_prompt = build_retry_prompt(original_prompt)
            
            response_text = call_llm(current_prompt, temperature=temperature)
            if not response_text:
                continue
            
            result = extract_json_from_response(response_text)
            if result is not None:
                final_result = result
                status = "成功" if attempt == 0 else "重试成功"
                success = True
                break
        
        if success and final_result:
            df.at[idx, '最终规范名称'] = final_result.get("final_name", "")
            df.at[idx, '审核结果'] = final_result.get("review_result", "")
            df.at[idx, '备注'] = final_result.get("remarks", "")
            df.at[idx, '审核状态'] = status
        else:
            df.at[idx, '审核状态'] = "解析失败"
            print(f"聚类 {cluster_id} 在 {max_parse_attempts} 次尝试后仍解析失败，已标记")
        
        if (idx + 1) % save_interval == 0:
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"文件处理完成，结果已保存至: {output_csv_path}")
    
    # 输出统计
    status_counts = df['审核状态'].value_counts()
    print("  处理状态统计:")
    for status, count in status_counts.items():
        print(f"    {status}: {count}")

if __name__ == "__main__":
    ensure_directory(STEP4_CONCLUSION_REVIEW_CSV.parent)

    file_pairs = [
        (STEP3_CONCLUSION_TEMPLATE_CSV, STEP4_CONCLUSION_REVIEW_CSV),
        (STEP3_IMPRESSION_TEMPLATE_CSV, STEP4_IMPRESSION_REVIEW_CSV),
    ]

    file_pairs = [(input_file, output_file) for input_file, output_file in file_pairs if input_file.exists()]

    if not file_pairs:
        print("没有可处理的文件，程序退出。")
        exit()

    print(f"找到 {len(file_pairs)} 个文件待处理")

    for input_file, output_file in tqdm(file_pairs, desc="总体处理进度", unit="文件"):
        llm_audit_clusters(str(input_file), str(output_file), save_interval=5, max_parse_attempts=3)
        print("-" * 50)

    print("所有文件处理完毕！")
