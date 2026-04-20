import pandas as pd
import requests
import os
import time
import logging
from tqdm import tqdm
from datetime import datetime
from pipeline_config import RAW_TEST_CSV, STEP1_OUTPUT_CSV, STEP1_LOG_FILE
from pipeline_utils import ensure_directory

# 配置API和文件路径
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
CSV_PATH = str(RAW_TEST_CSV)
OUTPUT_PATH = str(STEP1_OUTPUT_CSV)
LOG_DIR = str(STEP1_LOG_FILE.parent)
LOG_PATH = str(STEP1_LOG_FILE)


def create_directory_if_not_exists(path):
    """如果目录不存在则创建它"""
    if not os.path.exists(path):
        try:
            ensure_directory(path)
            print(f"已创建目录: {path}")
            return True
        except Exception as e:
            print(f"创建目录失败: {path} - {str(e)}")
            return False
    return True


def setup_logger():
    """配置日志记录器"""
    if not create_directory_if_not_exists(LOG_DIR):
        print(f"无法创建日志目录: {LOG_DIR}")
        return None

    logger = logging.getLogger("feature_extraction")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    try:
        file_handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
    except Exception as e:
        print(f"初始化日志失败: {str(e)}")
        return None


FEATURE_PROMPT_TEMPLATE = """
你是一个医疗报告特征提取助手。任务是将输入的患者检验报告文本，结合检查方式，分解为若干个使用分号分隔的元描述。每个元描述必须：
- 是一个相对完整的短语，描述一个独立的临床发现。
- 保留原始文本的所有关键信息，包括解剖位置、异常和量化值，部分原文本没有明确的位置信息，需要你进行保守地合理推断，并且使用#标明你推测的内容，请将位置信息放在每个元描述的最前方。
- 解剖位置应该为脊柱的某个解剖部位，或某些解剖部位，也可能根据检查方式和检验报告无法做出有效推测，请使用*将这些文本标注。
- 脊柱包含了颈椎，胸椎，腰椎，骶椎，尾椎，为了后续统一处理我希望将这些解剖部位信息完全替换成英文，仅使用大写字母C替代颈，T替代胸，L替代腰，S替代骶，W替代尾。
- 输出仅为分号分隔的字符串，无前缀、后缀或额外解释。

示例1：
检查方式：“颈椎MR检查(平扫)”
报告文本：“颈椎曲度及顺列欠佳，部分椎体边缘及椎小关节骨质增生，C4-6椎间隙变窄。,诸椎间盘信号减低，C4-7椎间盘不同程度突出，相应硬膜囊及脊髓受压，继发椎管狭窄。C4-7水平颈髓内见片状长T2信号影。,C4椎体后上方见小类圆形短T1长T2信号，压脂呈稍高信号。副鼻窦粘膜增厚。”
输出：“C椎曲度欠佳；C椎顺列欠佳；C椎部分椎体边缘及椎小关节骨质增生；C4-6椎间隙变窄；C椎诸椎间盘信号减低；C4-7椎间盘不同程度突出；C4-7相应硬膜囊及脊髓受压；C4-7继发椎管狭窄；C4-7水平颈髓内见片状长T2信号影；C4椎体后上方见小类圆形短T1长T2信号；C4椎体后上方压脂呈稍高信号；副鼻窦粘膜增厚。”

示例2：
检查方式：“全脊柱正侧位”
报告文本：“颈椎曲度直，顺列可，诸椎体缘及小关节骨质增生硬化，椎间隙未见明显狭窄。,胸椎曲度、顺列可，诸椎体缘及小关节骨质增生硬化，椎间隙未见明显狭窄。,腰椎侧弯，曲度、顺列可，诸椎体缘及小关节骨质增生硬化，椎间隙未见明显狭窄。,骶尾椎顺列可，诸椎体骨质增生。”
输出：“C椎曲度直；C椎顺列可；C椎诸椎体缘及小关节骨质增生硬化；C椎椎间隙未见明显狭窄；T椎曲度可；T椎顺列可；T椎诸椎体缘及小关节骨质增生硬化；T椎椎间隙未见明显狭窄；L椎侧弯；L椎曲度可；L椎顺列可；L椎诸椎体缘及小关节骨质增生硬化；L椎间隙未见明显狭窄；S椎顺列可；S椎诸椎体骨质增生。”

示例3：
检查方式：“全脊柱正侧位”
报告文本：“脊柱侧弯，颈椎曲度直，胸椎、腰椎、骶尾椎曲度可，椎体顺列可，椎间隙未见明显狭窄，部分椎体缘及椎小关节骨质增生硬化。L2楔形变。”
输出：“脊柱侧弯；C椎曲度直；T椎曲度可；L椎曲度可；S椎曲度可；W椎曲度可；C椎椎体顺列可；T椎椎体顺列可；S椎椎体顺列可；W椎椎体顺列可；C椎椎间隙未见明显狭窄；T椎椎间隙未见明显狭窄；L椎椎间隙未见明显狭窄；S椎椎间隙未见明显狭窄；W椎椎间隙未见明显狭窄；#脊柱#部分椎体缘及椎小关节骨质增生硬化；L2楔形变。”

现在，处理以下输入文本：
检查方式：“{method}”
输入文本：“{text}”
"""


def extract_features(method, text, logger, field_name=""):
    """调用ollama API提取诊断特征并记录耗时"""
    if not text or pd.isna(text) or not method or pd.isna(method):
        return "", 0

    try:
        start_time = time.time()

        prompt = FEATURE_PROMPT_TEMPLATE.format(method=method, text=text)
        data = {
            "model": "gpt-oss:120b",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1,
            "top_p": 0.5
        }

        response = requests.post(
            OLLAMA_API_URL,
            json=data,
            timeout=600
        )

        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            log_msg = (
                f"特征提取成功 | 字段: {field_name} | "
                f"耗时: {elapsed_time:.2f}秒 | 字符数: {len(text)}"
            )
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)
            return response_text, elapsed_time
        else:
            error_msg = f"API请求失败: {response.status_code} - {response.text[:100]}"
            log_msg = f"{error_msg} | 字段: {field_name} | 耗时: {elapsed_time:.2f}秒"
            if logger:
                logger.error(log_msg)
            else:
                print(log_msg)
            return "", elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        error_msg = f"处理过程中出错: {str(e)}"
        log_msg = f"{error_msg} | 字段: {field_name} | 耗时: {elapsed_time:.2f}秒"
        if logger:
            logger.error(log_msg)
        else:
            print(log_msg)
        return "", elapsed_time


def process_single_field(df, index, method, text, logger, source_col, result_col, time_col, status_col):
    """处理单个字段并写回DataFrame"""
    if not method or not text:
        df.at[index, result_col] = ""
        df.at[index, time_col] = 0.0
        df.at[index, status_col] = "空值"
        return False, 0.0

    features, proc_time = extract_features(method, text, logger, source_col)
    df.at[index, result_col] = features
    df.at[index, time_col] = round(proc_time, 2)
    df.at[index, status_col] = "成功" if features else "失败"

    return bool(features), proc_time


def process_csv(logger):
    """处理CSV文件主函数"""
    start_total = time.time()

    if logger:
        logger.info(f"====== 开始处理CSV文件: {CSV_PATH} ======")
    else:
        print(f"====== 开始处理CSV文件: {CSV_PATH} ======")

    if not os.path.exists(CSV_PATH):
        error_msg = f"CSV文件不存在: {CSV_PATH}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return

    output_dir = os.path.dirname(OUTPUT_PATH)
    if not create_directory_if_not_exists(output_dir):
        error_msg = f"无法创建输出目录: {output_dir}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return

    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
        msg = f"成功读取CSV文件，共 {len(df)} 条记录"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    except Exception as e:
        error_msg = f"读取CSV文件失败: {str(e)}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return

    # 检查必要列
    required_columns = ['检查方法', '诊断印象', '诊断结论']
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        error_msg = f"CSV文件中缺少必要列: {', '.join(missing_cols)}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return

    # 初始化输出列
    output_columns = {
        '诊断印象_特征提取结果': "",
        '诊断印象_特征提取耗时(秒)': 0.0,
        '诊断印象_特征提取状态': "未处理",
        '诊断结论_特征提取结果': "",
        '诊断结论_特征提取耗时(秒)': 0.0,
        '诊断结论_特征提取状态': "未处理"
    }

    for col, default_value in output_columns.items():
        if col not in df.columns:
            df[col] = default_value

    # 统计信息
    success_count_impression = 0
    success_count_conclusion = 0
    total_time_impression = 0.0
    total_time_conclusion = 0.0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="处理报告中"):
        method = str(row['检查方法']).strip() if pd.notna(row['检查方法']) else ""

        # 处理诊断印象
        text_impression = str(row['诊断印象']).strip() if pd.notna(row['诊断印象']) else ""
        success, proc_time = process_single_field(
            df=df,
            index=index,
            method=method,
            text=text_impression,
            logger=logger,
            source_col="诊断印象",
            result_col='诊断印象_特征提取结果',
            time_col='诊断印象_特征提取耗时(秒)',
            status_col='诊断印象_特征提取状态'
        )
        if success:
            success_count_impression += 1
            total_time_impression += proc_time

        # 处理诊断结论
        text_conclusion = str(row['诊断结论']).strip() if pd.notna(row['诊断结论']) else ""
        success, proc_time = process_single_field(
            df=df,
            index=index,
            method=method,
            text=text_conclusion,
            logger=logger,
            source_col="诊断结论",
            result_col='诊断结论_特征提取结果',
            time_col='诊断结论_特征提取耗时(秒)',
            status_col='诊断结论_特征提取状态'
        )
        if success:
            success_count_conclusion += 1
            total_time_conclusion += proc_time

        # 每处理5行保存一次
        if index % 5 == 0:
            try:
                df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
                time.sleep(0.5)
            except Exception as e:
                error_msg = f"保存进度失败: {str(e)}"
                if logger:
                    logger.error(error_msg)
                else:
                    print(error_msg)

    # 最终保存
    try:
        df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        msg = f"结果已保存至: {OUTPUT_PATH}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    except Exception as e:
        error_msg = f"保存最终结果失败: {str(e)}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)

    total_elapsed = time.time() - start_total
    avg_time_impression = (
        total_time_impression / success_count_impression
        if success_count_impression > 0 else 0
    )
    avg_time_conclusion = (
        total_time_conclusion / success_count_conclusion
        if success_count_conclusion > 0 else 0
    )

    summary = f"""
====== 处理完成! ======
总记录数: {len(df)}
诊断印象特征提取成功: {success_count_impression} 条
诊断印象特征提取平均处理时间: {avg_time_impression:.2f} 秒/条
诊断结论特征提取成功: {success_count_conclusion} 条
诊断结论特征提取平均处理时间: {avg_time_conclusion:.2f} 秒/条
总处理时间: {total_elapsed:.2f} 秒
结果已保存至: {OUTPUT_PATH}
日志已保存至: {LOG_PATH if logger else '无日志'}
"""

    if logger:
        logger.info(summary)
    else:
        print(summary)

    print(f"\n处理完成! 结果已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    logger = setup_logger()

    if not logger:
        print("警告: 日志功能未启用，所有信息将输出到控制台")

    process_csv(logger)
