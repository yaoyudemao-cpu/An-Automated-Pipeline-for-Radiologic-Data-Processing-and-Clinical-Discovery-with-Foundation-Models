# 本程序主要用于标注解剖部位

import pandas as pd
import re
import os
from pipeline_config import (
    ANATOMICAL_SITE_CONCLUSION_COL,
    ANATOMICAL_SITE_IMPRESSION_COL,
    STEP1_OUTPUT_CSV,
    STEP2_OUTPUT_CSV,
    STRUCTURED_CONCLUSION_COL,
    STRUCTURED_IMPRESSION_COL,
)
from pipeline_utils import ensure_directory, read_csv_with_fallback, require_columns

def preprocess_text(text):
    """数据预处理：替换文字、统一符号"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 替换文字
    replacements = {
        '颈': 'C',
        '胸': 'T', 
        '腰': 'L',
        '骶': 'S',
        '尾': 'W'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # 统一横线符号（将各种横线统一为-）
    text = re.sub(r'[—－–−─]', '-', text)
    
    return text

def is_t1_t2_signal(text, match_end):
    """判断是否为T1/T2信号，而不是解剖部位"""
    if match_end >= len(text):
        return False
    
    remaining_text = text[match_end:]
    return '信号' in remaining_text

def extract_segment(text):
    """从文本中提取解剖部位（改进版，处理更多情况）"""
    # 预处理文本，移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 定义解剖部位字母
    segment_letters = ['C', 'T', 'L', 'S', 'W']
    
    # 模式1: #...# 包裹的解剖部位（如#脊柱#）
    pattern_hash = r'#([^#]+?)#'
    hash_match = re.search(pattern_hash, text)
    if hash_match:
        segment = hash_match.group(1)
        # 检查这个解剖部位是否包含我们关心的椎体信息
        if any(char in segment for char in segment_letters + ['椎', '脊柱']):
            return segment
    
    # 模式2: %c%n-%c%n (如C3-C7)
    pattern1 = r'([CTLSW])\s*(\d{1,2})\s*-\s*([CTLSW])\s*(\d{1,2})'
    match1 = re.search(pattern1, text)
    if match1:
        # 检查是否为T1/T2信号
        if not (match1.group(1) == 'T' and match1.group(2) in ['1', '2'] and is_t1_t2_signal(text, match1.end())):
            segment = f"{match1.group(1)}{match1.group(2)}-{match1.group(3)}{match1.group(4)}"
            return segment
    
    # 模式3: %c%n-%n (如C3-7)
    pattern2 = r'([CTLSW])\s*(\d{1,2})\s*-\s*(\d{1,2})'
    match2 = re.search(pattern2, text)
    if match2:
        # 检查是否为T1/T2信号
        if not (match2.group(1) == 'T' and match2.group(2) in ['1', '2'] and is_t1_t2_signal(text, match2.end())):
            segment = f"{match2.group(1)}{match2.group(2)}-{match2.group(3)}"
            return segment
    
    # 模式4: %c-%c (如T-L)
    pattern3 = r'([CTLSW])\s*-\s*([CTLSW])'
    match3 = re.search(pattern3, text)
    if match3:
        segment = f"{match3.group(1)}-{match3.group(2)}"
        return segment
    
    # 模式5: %c%n (如C3)
    pattern4 = r'([CTLSW])\s*(\d{1,2})'
    match4 = re.search(pattern4, text)
    if match4:
        # 检查是否为T1/T2信号
        if not (match4.group(1) == 'T' and match4.group(2) in ['1', '2'] and is_t1_t2_signal(text, match4.end())):
            segment = f"{match4.group(1)}{match4.group(2)}"
            return segment
    
    # 模式6: %c椎 (如C椎)
    pattern5 = r'([CTLSW])\s*椎'
    match5 = re.search(pattern5, text)
    if match5:
        segment = f"{match5.group(1)}椎"
        return segment
    
    # 模式7: 单个字母%c (如C)
    pattern6 = r'\b([CTLSW])\b(?!\d)'
    match6 = re.search(pattern6, text)
    if match6:
        segment = match6.group(1)
        return segment
    
    # 模式8: 脊柱
    if '脊柱' in text:
        return '脊柱'
    
    return None

def process_finding(finding):
    """处理单个影像发现"""
    if pd.isna(finding) or finding == "":
        return ""
    
    # 保存原始文本用于调试
    original_finding = finding
    
    finding = preprocess_text(finding)
    segment = extract_segment(finding)
    
    if segment:
        # 使用更灵活的方法移除解剖部位部分
        # 首先尝试直接匹配解剖部位
        segment_pattern = re.escape(segment)
        problem = re.sub(segment_pattern, '', finding, count=1)
        
        # 如果直接替换没有效果，尝试更宽松的匹配
        if problem == finding:
            # 尝试匹配带有空格的变体
            segment_variants = [
                segment,
                segment.replace('-', ' - '),
                segment.replace('-', '- '),
                segment.replace('-', ' -')
            ]
            
            for variant in segment_variants:
                variant_pattern = re.escape(variant)
                problem = re.sub(variant_pattern, '', finding, count=1)
                if problem != finding:
                    break
        
        # 如果还是没有效果，尝试匹配#号包裹的解剖部位
        if problem == finding and '#' in finding:
            hash_pattern = f"#{re.escape(segment)}#"
            problem = re.sub(hash_pattern, '', finding, count=1)
        
        # 清理多余的空格和标点
        problem = re.sub(r'^[;\s，、]+', '', problem)
        problem = re.sub(r'[;\s，、]+$', '', problem)
        problem = problem.strip()
        
        # 移除问题中的星号和井号
        problem = problem.replace('*', '').replace('#', '')
        
        if problem:
            return f"{segment}@{problem}"
        else:
            return segment
    else:
        # 如果没有找到解剖部位，返回原始文本（经过预处理）
        result = finding
        # 移除星号和井号
        result = result.replace('*', '').replace('#', '')
        return result

def process_column(column):
    """处理整列数据"""
    processed_results = []
    
    for i, item in enumerate(column):
        if pd.isna(item) or item == "":
            processed_results.append("")
            continue
        
        # 分割影像发现（支持半角和全角分号）
        findings = re.split(r'[;；]', str(item))
        processed_findings = []
        
        for finding in findings:
            finding = finding.strip()
            if finding:
                processed_finding = process_finding(finding)
                if processed_finding:
                    processed_findings.append(processed_finding)
        
        # 用全角分号连接
        result = '；'.join(processed_findings)
        processed_results.append(result)
    
    return processed_results

def filter_examination_method(df):
    """过滤掉检查方法为'头颈CTA'和'头颈CTA+颅脑CTP'的记录"""
    if '检查方法' not in df.columns:
        print("警告：数据框中没有'检查方法'列，跳过过滤步骤")
        return df, 0
    
    # 记录过滤前的行数
    original_count = len(df)
    
    # 定义需要过滤的检查方法（考虑可能的空格变体）
    methods_to_filter = [
        '头颈CTA',
        '头颈CTA+颅脑CTP',
        '头颈 CTA',
        '头颈 CTA + 颅脑 CTP',
        '头颈CTA + 颅脑CTP'
    ]
    
    # 过滤数据（使用更宽松的匹配）
    def should_filter(method):
        if pd.isna(method):
            return False
        method_str = str(method).strip()
        # 移除所有空格后比较
        method_no_space = re.sub(r'\s+', '', method_str)
        for pattern in methods_to_filter:
            pattern_no_space = re.sub(r'\s+', '', pattern)
            if method_no_space == pattern_no_space:
                return True
        return False
    
    filtered_df = df[~df['检查方法'].apply(should_filter)]
    
    # 计算过滤掉的行数
    filtered_count = original_count - len(filtered_df)
    
    print(f"过滤掉 {filtered_count} 条记录（检查方法为'头颈CTA'或'头颈CTA+颅脑CTP'）")
    print(f"过滤后剩余 {len(filtered_df)} 条记录")
    
    return filtered_df, filtered_count

def main():
    input_file = STEP1_OUTPUT_CSV
    output_file = STEP2_OUTPUT_CSV

    if not input_file.exists():
        print(f"文件不存在: {input_file}")
        return

    ensure_directory(output_file.parent)

    print(f"\n正在处理文件: {input_file.name}")

    try:
        df = read_csv_with_fallback(input_file)
        print(f"读取到 {len(df)} 条记录")

        impression_col = STRUCTURED_IMPRESSION_COL
        conclusion_col = STRUCTURED_CONCLUSION_COL

        require_columns(df, [impression_col, conclusion_col], input_file.name)

        df, filtered_count = filter_examination_method(df)

        if len(df) == 0:
            print(f"文件 {input_file.name} 过滤后没有数据，跳过处理")
            return

        print(f"处理'{impression_col}'列...")
        df[ANATOMICAL_SITE_IMPRESSION_COL] = process_column(df[impression_col])

        print(f"处理'{conclusion_col}'列...")
        df[ANATOMICAL_SITE_CONCLUSION_COL] = process_column(df[conclusion_col])

        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"处理完成，结果已保存为: {output_file}")

        print("\n前3行处理结果示例:")
        print(df[[impression_col, ANATOMICAL_SITE_IMPRESSION_COL, conclusion_col, ANATOMICAL_SITE_CONCLUSION_COL]].head(3))
        print("-" * 50)
        print(f"\n总共过滤掉 {filtered_count} 条记录")

    except Exception as e:
        print(f"处理文件 {input_file.name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
