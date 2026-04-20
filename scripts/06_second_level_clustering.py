#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断名词自动分类与审核系统（增强版）
- 聚类阶段提示词约束（方案B）：优先使用现有分类，控制新类别创建
- 审核迭代后增加合并阶段（方案A）：自动合并相似类别，减少冗余
- 自动格式审查与重试，每批/每轮保存中间结果
- 分别处理诊断印象和诊断结论两列
- 所有输出保存在带时间戳的目录中
"""

import os
import json
import time
import random
import logging
import requests
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from datetime import datetime
from pipeline_config import (
    NORMALIZED_CONCLUSION_COL,
    NORMALIZED_IMPRESSION_COL,
    STEP5_OUTPUT_CSV,
    STEP6_LOG_FILE,
    STEP6_OUTPUT_DIR,
)
from pipeline_utils import ensure_directory, read_csv_with_fallback, require_columns

# ==================== 配置 ====================
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "gpt-oss:120b"          # 根据实际情况修改
TEMPERATURE = 0.1
TOP_P = 0.5
TIMEOUT = 600                        # 秒
MAX_RETRIES = 3                      # 格式错误最大重试次数
RETRY_DELAY = 2                       # 重试前等待秒数
MERGE_THRESHOLD = 60                  # 如果类别数量超过此值，触发合并（可调整）

# 先验分类列表（可根据需要修改）
PREDEFINED_CATEGORIES = [
    
]

# 全局变量：输出目录（由主函数创建）
OUTPUT_DIR = None
logger = logging.getLogger(__name__)

# ==================== 增强JSON提取函数 ====================
def extract_json(text: str) -> Optional[str]:
    """
    从文本中提取第一个完整的JSON对象或数组。
    使用栈匹配最外层大括号/中括号，并验证是否为有效JSON。
    """
    # 先尝试找数组
    stack = []
    start = -1
    for i, ch in enumerate(text):
        if ch == '[':
            if not stack:
                start = i
            stack.append(ch)
        elif ch == ']':
            if stack:
                stack.pop()
                if not stack and start != -1:
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        pass
    # 如果没有数组，再找对象
    stack = []
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start = i
            stack.append(ch)
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start != -1:
                    candidate = text[start:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        pass
    return None

# ==================== LLM调用函数（带重试和格式检查）====================
def call_llm_with_retry(prompt: str, expected_format: str = "json", max_retries: int = MAX_RETRIES,
                        context: str = "") -> Optional[str]:
    """
    调用LLM，如果返回内容不符合预期格式则重试
    :param prompt: 提示词
    :param expected_format: 期望格式，目前仅支持"json"
    :param max_retries: 最大重试次数
    :param context: 额外上下文信息（用于记录失败响应）
    :return: 符合格式的响应文本，或None
    """
    for attempt in range(1, max_retries + 1):
        logger.debug(f"LLM调用尝试 {attempt}/{max_retries} - {context}")
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P
                },
                timeout=TIMEOUT
            )
            if response.status_code != 200:
                logger.warning(f"HTTP错误 {response.status_code}: {response.text[:200]}")
                time.sleep(RETRY_DELAY)
                continue

            text = response.json().get("response", "").strip()
            if not text:
                logger.warning("LLM返回空文本")
                time.sleep(RETRY_DELAY)
                continue

            if expected_format == "json":
                if extract_json(text) is not None:
                    return text
                else:
                    logger.warning(f"无法提取有效JSON，响应片段: {text[:200]}")
                    if OUTPUT_DIR:
                        fail_file = os.path.join(OUTPUT_DIR, "failed_responses.log")
                        with open(fail_file, "a", encoding="utf-8") as f:
                            f.write(f"\n--- {datetime.now()} - {context} (尝试 {attempt}) ---\n")
                            f.write(text)
                            f.write("\n--- 结束 ---\n")
                    time.sleep(RETRY_DELAY)
                    continue
            else:
                return text

        except requests.exceptions.RequestException as e:
            logger.warning(f"请求异常: {e}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.warning(f"未知异常: {e}")
            time.sleep(RETRY_DELAY)

    logger.error(f"达到最大重试次数 {max_retries}，放弃该请求 - {context}")
    return None

# ==================== Prompt模板 ====================
# 方案B：在聚类提示词中增加约束
CLUSTERING_PROMPT_TEMPLATE = """
你是一个医学影像诊断分类专家。现有一些诊断名词，需要根据给定的分类体系进行分类。你的任务：
1. 为每个诊断名词分配一个或多个最合适的分类（从现有分类中选择）。
2. 请优先使用现有分类。只有当某个诊断无法合理归入任何现有分类时，才创建新分类。
   创建的新分类名称应简洁、概括，避免与现有分类重复，且不要过于具体。
3. 输出格式必须为严格的JSON数组，每个元素是一个对象，包含：
   - "diagnosis": 原始诊断名词
   - "categories": 分类列表（字符串数组），如果新建了分类，新分类名称也在此列表中
   - "new_categories": 仅当新建分类时才包含此字段，值为新建分类名称列表（如有多个新分类）
   注意：不要添加任何额外解释或标记。

现有分类列表（用双引号括起，以逗号分隔）：
{category_list}

诊断名词列表（每个诊断一行）：
{diagnosis_list}

请直接输出JSON数组。
"""

REVIEW_PROMPT_TEMPLATE = """
你是一个医学影像诊断分类质量审核员。现有分类 "{category}" 包含以下诊断名词：
{diagnosis_items}

请审核该分类是否合适，并回答以下问题：
1. 这些诊断是否都适合归入 "{category}"？如果不适合，请指出哪些诊断应当移出，并说明原因。
2. 是否存在某些诊断实际上应归入其他现有分类？如果是，请指出诊断名词及建议的分类。
3. 是否存在某些诊断需要创建新的分类？如果是，请指出诊断名词及建议的新分类名称。

请以JSON格式输出，**必须是一个对象**，包含以下字段：
- "is_appropriate": true/false，表示该分类整体是否合适。
- "misclassified": [{{"diagnosis": "...", "reason": "..."}}]，列出应移出的诊断及原因。
- "suggested_moves": [{{"diagnosis": "...", "target_category": "..."}}]，建议移入其他现有分类的诊断。
- "suggested_new_categories": [{{"diagnosis": "...", "new_category": "..."}}]，建议新建分类的诊断及新分类名称。

只输出JSON对象，不要输出数组或其他格式。
"""

# 方案A：类别合并提示词
MERGE_PROMPT_TEMPLATE = """
你是一个医学影像诊断分类专家。现有以下分类列表（每个分类是一个诊断名词的类别）：
{category_list}

请检查这些分类，找出语义相似、可以合并的类别。合并原则：
- 如果两个或多个分类描述的是同一病理现象，只是表述略有不同，可以合并为一个更通用的分类。
- 合并后的分类名称应能涵盖所有被合并的分类，保持临床准确性。
- 如果一个分类是另一个分类的子集，可以考虑合并或保留（请说明理由）。
- 仅合并那些明确可合并的，不确定的保留原样。

请以JSON格式输出一个合并映射，格式为：
{{
    "合并后类别名称1": ["原类别1", "原类别2", ...],
    "合并后类别名称2": ["原类别3", "原类别4", ...],
    ...
}}
只输出JSON对象，不要包含其他内容。
"""

# ==================== 解析函数 ====================
def parse_clustering_response(response_text: str) -> List[Dict]:
    """解析聚类返回的JSON，期望得到数组"""
    json_str = extract_json(response_text)
    if not json_str:
        logger.error("未找到有效的JSON")
        return []
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            # 尝试从对象中提取数组字段
            for key in ["result", "results", "data", "diagnoses", "items"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            logger.error(f"返回的是对象但不是预期的数组格式: {list(data.keys())}")
            return []
        elif isinstance(data, list):
            return data
        else:
            logger.error(f"返回数据不是列表也不是对象: {type(data)}")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        return []

def parse_review_response(response_text: str) -> Dict:
    """
    解析审核返回的JSON。
    期望得到对象，但如果返回的是数组，尝试将其转换为合理的对象。
    """
    json_str = extract_json(response_text)
    if not json_str:
        logger.error("未找到有效的JSON对象")
        return {}
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            # 如果返回的是数组，尝试将其转换为审核对象
            logger.warning(f"审核返回的是数组，长度 {len(data)}，尝试转换为对象")
            misclassified = []
            for item in data:
                if isinstance(item, dict):
                    diag = item.get("diagnosis")
                    if diag:
                        misclassified.append({"diagnosis": diag, "reason": item.get("reason", "未提供原因")})
                elif isinstance(item, str):
                    misclassified.append({"diagnosis": item, "reason": "可能误分类"})
            return {
                "is_appropriate": False,
                "misclassified": misclassified,
                "suggested_moves": [],
                "suggested_new_categories": []
            }
        else:
            logger.error(f"返回数据不是对象也不是数组: {type(data)}")
            return {}
    except json.JSONDecodeError as e:
        logger.error(f"审核JSON解析失败: {e}")
        return {}

def parse_merge_response(response_text: str) -> Dict[str, List[str]]:
    """解析合并映射的JSON，期望得到一个字典"""
    json_str = extract_json(response_text)
    if not json_str:
        logger.error("未找到有效的合并映射JSON")
        return {}
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data
        else:
            logger.error(f"合并映射返回的不是对象: {type(data)}")
            return {}
    except json.JSONDecodeError as e:
        logger.error(f"合并映射JSON解析失败: {e}")
        return {}

# ==================== 核心分类器类 ====================
class DiagnosisClassifier:
    def __init__(self, diagnoses: List[str], initial_categories: List[str], output_prefix: str):
        self.diagnoses = diagnoses
        self.categories = set(initial_categories)
        self.result = {}          # diagnosis -> set of categories
        self.new_categories_log = []  # 记录所有新建的分类名称
        self.output_prefix = output_prefix
        self.output_dir = OUTPUT_DIR

    def _build_category_list(self) -> str:
        return ', '.join(f'"{c}"' for c in sorted(self.categories))

    def _get_file_path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename) if self.output_dir else filename

    def cluster_batch(self, batch_diagnoses: List[str], batch_num: int) -> Tuple[Dict[str, Set[str]], Set[str]]:
        context = f"{self.output_prefix}_batch{batch_num}"
        prompt = CLUSTERING_PROMPT_TEMPLATE.format(
            category_list=self._build_category_list(),
            diagnosis_list='\n'.join(batch_diagnoses)
        )
        response = call_llm_with_retry(prompt, expected_format="json", context=context)
        if not response:
            logger.warning(f"{context} - LLM返回空，跳过该批")
            return {}, set()

        parsed = parse_clustering_response(response)
        if not parsed:
            logger.warning(f"{context} - 解析失败，跳过该批")
            return {}, set()

        batch_result = {}
        new_cats = set()
        for item in parsed:
            diag = item.get("diagnosis", "").strip()
            if not diag:
                continue
            cats = item.get("categories", [])
            for cat in cats:
                if cat not in self.categories:
                    new_cats.add(cat)
            batch_result[diag] = set(cats)
        return batch_result, new_cats

    def run_clustering(self, batch_size_ratio: float = 0.05, shuffle: bool = True, seed: int = 42):
        logger.info("开始聚类2a过程...")
        diagnoses_copy = self.diagnoses.copy()
        if shuffle:
            random.seed(seed)
            random.shuffle(diagnoses_copy)

        batch_size = max(1, int(len(diagnoses_copy) * batch_size_ratio))
        total = len(diagnoses_copy)
        logger.info(f"总诊断数: {total}, 批次大小: {batch_size}")

        for i in range(0, total, batch_size):
            batch = diagnoses_copy[i:i+batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"处理批次 {batch_num}/{(total-1)//batch_size + 1} ...")
            batch_result, new_cats = self.cluster_batch(batch, batch_num)

            self.categories.update(new_cats)
            if new_cats:
                logger.info(f"新增分类: {new_cats}")
                self.new_categories_log.extend(list(new_cats))

            for diag, cats in batch_result.items():
                if diag in self.result:
                    self.result[diag].update(cats)
                else:
                    self.result[diag] = cats

            intermediate_df = self.to_dataframe()
            intermediate_file = self._get_file_path(f"{self.output_prefix}_batch_{batch_num}_intermediate.csv")
            intermediate_df.to_csv(intermediate_file, index=False, encoding='utf-8-sig')
            logger.info(f"批次中间结果已保存: {intermediate_file}")

            time.sleep(1)

        logger.info("聚类2a完成。")
        logger.info(f"最终分类数量: {len(self.categories)}")
        logger.info(f"新建分类: {self.new_categories_log}")

    def review_all_categories(self, round_num: int = 1) -> Set[str]:
        logger.info("开始分类审核...")
        suspicious_diagnoses = set()

        cat_to_diagnoses = defaultdict(set)
        for diag, cats in self.result.items():
            for cat in cats:
                cat_to_diagnoses[cat].add(diag)

        for category, diagnoses_set in cat_to_diagnoses.items():
            if not diagnoses_set:
                continue
            logger.info(f"审核分类: {category} (包含 {len(diagnoses_set)} 个诊断)")
            diag_list_str = '\n'.join(f"- {d}" for d in sorted(diagnoses_set))
            prompt = REVIEW_PROMPT_TEMPLATE.format(
                category=category,
                diagnosis_items=diag_list_str
            )
            context = f"{self.output_prefix}_review_{category}"
            response = call_llm_with_retry(prompt, expected_format="json", context=context)
            if not response:
                logger.warning(f"分类 {category} 审核无响应，跳过")
                continue
            review = parse_review_response(response)
            if not review:
                continue

            for item in review.get("misclassified", []):
                diag = item.get("diagnosis")
                if diag:
                    suspicious_diagnoses.add(diag)
            for item in review.get("suggested_moves", []):
                diag = item.get("diagnosis")
                if diag:
                    suspicious_diagnoses.add(diag)
            for item in review.get("suggested_new_categories", []):
                diag = item.get("diagnosis")
                if diag:
                    suspicious_diagnoses.add(diag)

        logger.info(f"审核完成，发现 {len(suspicious_diagnoses)} 个可疑诊断")
        return suspicious_diagnoses

    def refine_with_review(self, max_review_rounds: int = 2):
        for round_num in range(max_review_rounds):
            logger.info(f"===== 审核迭代第 {round_num+1} 轮 =====")
            pre_df = self.to_dataframe()
            pre_file = self._get_file_path(f"{self.output_prefix}_review_round{round_num+1}_pre.csv")
            pre_df.to_csv(pre_file, index=False, encoding='utf-8-sig')
            logger.info(f"审核前状态已保存: {pre_file}")

            suspicious = self.review_all_categories(round_num+1)
            if not suspicious:
                logger.info("无可疑诊断，审核通过。")
                break

            logger.info(f"对 {len(suspicious)} 个可疑诊断进行重新聚类...")
            for diag in suspicious:
                if diag in self.result:
                    del self.result[diag]

            suspicious_list = list(suspicious)
            batch_result, new_cats = self.cluster_batch(suspicious_list, f"review_round{round_num+1}")
            self.categories.update(new_cats)
            if new_cats:
                logger.info(f"新增分类: {new_cats}")
                self.new_categories_log.extend(list(new_cats))
            for diag, cats in batch_result.items():
                self.result[diag] = cats

            post_df = self.to_dataframe()
            post_file = self._get_file_path(f"{self.output_prefix}_review_round{round_num+1}_post.csv")
            post_df.to_csv(post_file, index=False, encoding='utf-8-sig')
            logger.info(f"审核后结果已保存: {post_file}")

        else:
            logger.warning("达到最大审核迭代次数，仍有部分诊断可能未妥善分类。")

    def merge_categories(self):
        """合并相似类别，减少冗余"""
        all_cats = sorted(self.categories)
        if len(all_cats) <= MERGE_THRESHOLD:
            logger.info(f"类别数量 {len(all_cats)} 小于等于阈值 {MERGE_THRESHOLD}，跳过合并")
            return

        logger.info(f"当前类别数量 {len(all_cats)}，超过阈值 {MERGE_THRESHOLD}，开始合并...")
        cat_list_str = '\n'.join(f"- {cat}" for cat in all_cats)
        prompt = MERGE_PROMPT_TEMPLATE.format(category_list=cat_list_str)
        context = f"{self.output_prefix}_merge"
        response = call_llm_with_retry(prompt, expected_format="json", context=context)
        if not response:
            logger.warning("合并无响应，跳过")
            return

        merge_map = parse_merge_response(response)
        if not merge_map:
            logger.warning("合并映射解析失败，跳过")
            return

        # 验证映射格式：每个键值对的值应为列表
        valid_map = {}
        for new_cat, old_cats in merge_map.items():
            if isinstance(old_cats, list):
                valid_map[new_cat] = old_cats
            else:
                logger.warning(f"合并映射中 {new_cat} 的值不是列表，忽略")

        if not valid_map:
            logger.warning("无有效的合并映射，跳过")
            return

        # 收集所有将被合并的原类别
        merged_old_cats = set()
        for old_cats in valid_map.values():
            merged_old_cats.update(old_cats)

        # 未合并的类别保留
        remaining_old_cats = set(all_cats) - merged_old_cats

        # 构建新类别集合
        new_categories = set(valid_map.keys()) | remaining_old_cats

        # 更新每个诊断的类别
        new_result = {}
        for diag, old_cat_set in self.result.items():
            new_cat_set = set()
            for old_cat in old_cat_set:
                # 查找旧类别属于哪个新类别
                found = False
                for new_cat, old_cats in valid_map.items():
                    if old_cat in old_cats:
                        new_cat_set.add(new_cat)
                        found = True
                        break
                if not found:
                    new_cat_set.add(old_cat)
            new_result[diag] = new_cat_set

        self.categories = new_categories
        self.result = new_result
        logger.info(f"合并完成，类别数量从 {len(all_cats)} 减少到 {len(new_categories)}")

    def to_dataframe(self) -> pd.DataFrame:
        all_cats = sorted(self.categories)
        rows = []
        for diag in self.diagnoses:
            cats = self.result.get(diag, set())
            row = {"诊断名词": diag}
            for cat in all_cats:
                row[cat] = 1 if cat in cats else 0
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

# ==================== 从文本列提取诊断名词 ====================
def extract_diagnoses_from_column(df: pd.DataFrame, column_name: str) -> List[str]:
    all_diag = set()
    for text in df[column_name].dropna():
        parts = str(text).split('；')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if '@' in part:
                _, diag = part.split('@', 1)
                diag = diag.strip()
            else:
                diag = part
            if diag:
                all_diag.add(diag)
    return sorted(all_diag)

# ==================== 处理单个列的函数 ====================
def process_column(df: pd.DataFrame, column_name: str, output_prefix: str):
    logger.info(f"开始处理列: {column_name}，输出前缀: {output_prefix}")
    diagnoses = extract_diagnoses_from_column(df, column_name)
    if not diagnoses:
        logger.warning(f"列 {column_name} 未提取到任何诊断名词，跳过")
        return
    logger.info(f"从列 {column_name} 提取到 {len(diagnoses)} 个唯一诊断名词")

    classifier = DiagnosisClassifier(diagnoses, PREDEFINED_CATEGORIES, output_prefix)

    # 执行聚类2a（每批后自动保存中间结果）
    classifier.run_clustering(batch_size_ratio=0.05, shuffle=True)

    # 执行审核与修正（每轮前后自动保存）
    classifier.refine_with_review(max_review_rounds=2)

    # 方案A：合并相似类别
    classifier.merge_categories()

    # 保存最终结果
    final_df = classifier.to_dataframe()
    final_excel = classifier._get_file_path(f"{output_prefix}_final_classification.xlsx")
    final_df.to_excel(final_excel, index=False)
    final_csv = classifier._get_file_path(f"{output_prefix}_final_classification.csv")
    final_df.to_csv(final_csv, index=False, encoding='utf-8-sig')
    logger.info(f"最终分类结果已保存至 {final_excel} 和 {final_csv}")

    # 保存新建分类列表
    new_cats_file = classifier._get_file_path(f"{output_prefix}_new_categories.log")
    with open(new_cats_file, 'w', encoding='utf-8') as f:
        for cat in classifier.new_categories_log:
            f.write(cat + '\n')
    logger.info(f"新建分类列表已保存至 {new_cats_file}")

    logger.info(f"列 {column_name} 处理完成。")

# ==================== 主程序 ====================
def main():
    global OUTPUT_DIR, logger

    input_file = STEP5_OUTPUT_CSV

    if not input_file.exists():
        print(f"输入文件不存在: {input_file}")
        return

    OUTPUT_DIR = str(STEP6_OUTPUT_DIR)
    ensure_directory(OUTPUT_DIR)
    print(f"输出目录已创建: {OUTPUT_DIR}")

    # 配置日志：同时输出到文件和控制台
    file_handler = logging.FileHandler(STEP6_LOG_FILE, encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"输出目录: {OUTPUT_DIR}")

    # 读取数据（使用UTF-8-SIG编码）
    try:
        df = read_csv_with_fallback(input_file)
        require_columns(
            df,
            [NORMALIZED_IMPRESSION_COL, NORMALIZED_CONCLUSION_COL],
            input_file.name,
        )
        logger.info(f"成功读取文件: {input_file}，共 {len(df)} 行")
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return

    # 定义要处理的列和对应的输出前缀
    columns_to_process = [
        (NORMALIZED_IMPRESSION_COL, "impression"),
        (NORMALIZED_CONCLUSION_COL, "conclusion"),
    ]

    for col_name, prefix in columns_to_process:
        if col_name not in df.columns:
            logger.error(f"列 {col_name} 不存在于文件中，跳过")
            continue
        process_column(df, col_name, prefix)
        logger.info(f"完成处理列 {col_name}，等待5秒后处理下一列...")
        time.sleep(5)

    logger.info("所有处理完成！")

if __name__ == "__main__":
    main()
