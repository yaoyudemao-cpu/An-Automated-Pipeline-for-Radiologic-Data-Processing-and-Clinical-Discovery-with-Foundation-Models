# diagnosis_processor_apply.py - 根据审核后的 CSV 应用规范化

import pandas as pd

from pipeline_config import (
    ANATOMICAL_SITE_CONCLUSION_COL,
    ANATOMICAL_SITE_IMPRESSION_COL,
    NORMALIZED_CONCLUSION_COL,
    NORMALIZED_IMPRESSION_COL,
    STEP2_OUTPUT_CSV,
    STEP5_CONCLUSION_REVIEW_CANDIDATES,
    STEP5_IMPRESSION_REVIEW_CANDIDATES,
    STEP5_OUTPUT_CSV,
    STEP5_UNMAPPED_CONCLUSION_TXT,
    STEP5_UNMAPPED_IMPRESSION_TXT,
)
from pipeline_utils import (
    ensure_directory,
    first_existing_path,
    read_csv_with_fallback,
    require_columns,
)


class SegmentDiagnosisProcessor:
    def __init__(self):
        self.raw_data = pd.DataFrame()

    def read_file(self, csv_path):
        """读取 CSV 文件"""
        df = read_csv_with_fallback(csv_path)
        df["source_file"] = csv_path.name
        self.raw_data = df
        print(f"成功读取文件: {csv_path.name}, 包含 {len(df)} 行数据")
        return df

    def extract_segment_diagnosis_items(self, source_col, count_col_name):
        """提取指定列中的所有诊断项"""
        print(f"提取 {source_col} 中的诊断项...")

        all_items = []

        for idx, row in self.raw_data.iterrows():
            text = row.get(source_col, "")
            if pd.isna(text) or text == "":
                continue

            parts = str(text).split("；")

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if "@" in part:
                    segment, diagnosis = part.split("@", 1)
                    segment = segment.strip()
                    diagnosis = diagnosis.strip()
                else:
                    segment = ""
                    diagnosis = part.strip()

                if diagnosis:
                    all_items.append(
                        {
                            "original_index": idx,
                            "source_file": row["source_file"],
                            "segment": segment,
                            "diagnosis": diagnosis,
                            "full_text": part,
                        }
                    )

        df_items = pd.DataFrame(all_items)

        if len(df_items) > 0:
            diagnosis_counts = df_items["diagnosis"].value_counts().reset_index()
            diagnosis_counts.columns = [count_col_name, "出现次数"]
        else:
            diagnosis_counts = pd.DataFrame(columns=[count_col_name, "出现次数"])

        print(f"共提取到 {len(all_items)} 个诊断项，{len(diagnosis_counts)} 个唯一{count_col_name}")
        return df_items, diagnosis_counts

    def create_hard_mapping_from_csv(self, reviewed_csv_file):
        """从审核后的 CSV 文件创建硬性映射表"""
        print(f"加载审核结果: {reviewed_csv_file}")
        mapping_df = read_csv_with_fallback(reviewed_csv_file)
        require_columns(mapping_df, ["聚类ID", "聚类内诊断结论", "推荐规范名称"], reviewed_csv_file.name)

        hard_mapping = {}

        for _, row in mapping_df.iterrows():
            cluster_id = row["聚类ID"]
            final_name = row["最终规范名称"] if pd.notnull(row.get("最终规范名称", "")) else ""
            review_result = row["审核结果"] if pd.notnull(row.get("审核结果", "")) else "接受"
            cluster_diagnoses_str = row["聚类内诊断结论"] if pd.notnull(row.get("聚类内诊断结论", "")) else ""

            if not final_name:
                final_name = row["推荐规范名称"] if pd.notnull(row.get("推荐规范名称", "")) else ""

            if not final_name:
                print(f"警告: 聚类ID {cluster_id} 的最终规范名称为空，跳过该聚类")
                continue

            if cluster_diagnoses_str:
                parts = cluster_diagnoses_str.split(";")

                for part in parts:
                    part = part.strip()
                    if not part:
                        continue

                    if "(" in part and ")" in part:
                        left_paren = part.rfind("(")
                        right_paren = part.rfind(")")
                        if left_paren < right_paren:
                            diagnosis = part[:left_paren].strip()
                        else:
                            diagnosis = part
                    else:
                        diagnosis = part

                    if review_result in ("接受", "修改", "拆分"):
                        hard_mapping[diagnosis] = final_name

        print(f"创建了 {len(hard_mapping)} 个映射")
        for index, (key, value) in enumerate(list(hard_mapping.items())[:10], start=1):
            print(f"  {index}. '{key}' -> '{value}'")

        return hard_mapping

    def apply_hard_mapping(self, df_items, hard_mapping, unmapped_output_path):
        """应用硬性映射到数据"""
        print("应用硬性映射到诊断项...")

        mapped_count = 0
        not_mapped_count = 0
        not_mapped_items = set()
        standardized_diagnoses = []

        for _, row in df_items.iterrows():
            diagnosis = row["diagnosis"]

            if diagnosis in hard_mapping:
                standardized = hard_mapping[diagnosis]
                mapped_count += 1
            else:
                diagnosis_trimmed = diagnosis.strip()
                if diagnosis_trimmed in hard_mapping:
                    standardized = hard_mapping[diagnosis_trimmed]
                    mapped_count += 1
                else:
                    standardized = diagnosis
                    not_mapped_count += 1
                    not_mapped_items.add(diagnosis)

            standardized_diagnoses.append(standardized)

        df_items["standardized_diagnosis"] = standardized_diagnoses

        print(f"映射成功: {mapped_count}, 未映射: {not_mapped_count}")

        if not_mapped_items:
            ensure_directory(unmapped_output_path.parent)
            with open(unmapped_output_path, "w", encoding="utf-8") as file_obj:
                for item in sorted(not_mapped_items):
                    file_obj.write(item + "\n")
            print(f"未映射诊断已保存到: {unmapped_output_path}")

        return df_items

    def reconstruct_original_format(self, df_items, total_rows):
        """重建原始格式（解剖部位@诊断）"""
        print("重建原始格式...")

        reconstructed_texts = [""] * total_rows
        if df_items.empty:
            return reconstructed_texts

        grouped = df_items.groupby("original_index")

        for idx, group in grouped:
            parts = []
            for _, row in group.iterrows():
                segment = row["segment"]
                standardized = row["standardized_diagnosis"]

                if segment:
                    parts.append(f"{segment}@{standardized}")
                else:
                    parts.append(standardized)

            if idx < total_rows:
                reconstructed_texts[idx] = "；".join(parts)

        return reconstructed_texts

    def save_final_results(self, df_original, reconstructed_conclusion_texts, reconstructed_impression_texts, output_file):
        """保存最终结果"""
        ensure_directory(output_file.parent)

        df_final = df_original.copy()
        df_final[NORMALIZED_CONCLUSION_COL] = reconstructed_conclusion_texts
        df_final[NORMALIZED_IMPRESSION_COL] = reconstructed_impression_texts
        df_final.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"规范化完成，结果保存在: {output_file}")

        check_pairs = [
            (ANATOMICAL_SITE_CONCLUSION_COL, NORMALIZED_CONCLUSION_COL),
            (ANATOMICAL_SITE_IMPRESSION_COL, NORMALIZED_IMPRESSION_COL),
        ]

        total_rows = len(df_final)
        for original_col, normalized_col in check_pairs:
            if original_col in df_final.columns and normalized_col in df_final.columns:
                changes = 0
                for _, row in df_final.iterrows():
                    original = row[original_col] if pd.notna(row[original_col]) else ""
                    normalized = row[normalized_col] if pd.notna(row[normalized_col]) else ""
                    if original != normalized:
                        changes += 1

                if total_rows > 0:
                    ratio = changes / total_rows * 100
                    print(f"{normalized_col} 变更数量: {changes}/{total_rows} ({ratio:.1f}%)")

        return df_final


def main():
    """从 CSV 文件应用规范化"""
    input_file = STEP2_OUTPUT_CSV

    if not input_file.exists():
        print(f"输入文件不存在: {input_file}")
        return

    conclusion_review_file = first_existing_path(STEP5_CONCLUSION_REVIEW_CANDIDATES)
    impression_review_file = first_existing_path(STEP5_IMPRESSION_REVIEW_CANDIDATES)

    if conclusion_review_file is None:
        print("未找到诊断结论审核文件，无法继续")
        return

    if impression_review_file is None:
        print("未找到诊断印象审核文件，无法继续")
        return

    processor = SegmentDiagnosisProcessor()

    try:
        print("步骤1: 读取原始数据...")
        data = processor.read_file(input_file)
        total_rows = len(data)
        require_columns(
            data,
            [ANATOMICAL_SITE_CONCLUSION_COL, ANATOMICAL_SITE_IMPRESSION_COL],
            input_file.name,
        )

        print("\n步骤2: 提取诊断结论项...")
        df_items_conclusion, _ = processor.extract_segment_diagnosis_items(
            source_col=ANATOMICAL_SITE_CONCLUSION_COL,
            count_col_name="诊断结论",
        )

        print("\n步骤2.1: 提取诊断印象项...")
        df_items_impression, _ = processor.extract_segment_diagnosis_items(
            source_col=ANATOMICAL_SITE_IMPRESSION_COL,
            count_col_name="诊断印象",
        )

        print("\n步骤3: 从 CSV 文件创建硬性映射...")
        hard_mapping_conclusion = processor.create_hard_mapping_from_csv(conclusion_review_file)
        hard_mapping_impression = processor.create_hard_mapping_from_csv(impression_review_file)

        print("\n步骤4: 应用硬性映射到诊断结论...")
        df_items_conclusion_mapped = processor.apply_hard_mapping(
            df_items_conclusion,
            hard_mapping_conclusion,
            STEP5_UNMAPPED_CONCLUSION_TXT,
        )

        print("\n步骤4.1: 应用硬性映射到诊断印象...")
        df_items_impression_mapped = processor.apply_hard_mapping(
            df_items_impression,
            hard_mapping_impression,
            STEP5_UNMAPPED_IMPRESSION_TXT,
        )

        print("\n步骤5: 重建原始格式...")
        reconstructed_conclusion_texts = processor.reconstruct_original_format(
            df_items_conclusion_mapped,
            total_rows,
        )
        reconstructed_impression_texts = processor.reconstruct_original_format(
            df_items_impression_mapped,
            total_rows,
        )

        print("\n步骤6: 保存结果...")
        df_final = processor.save_final_results(
            data,
            reconstructed_conclusion_texts,
            reconstructed_impression_texts,
            STEP5_OUTPUT_CSV,
        )

        print("\n规范化处理完成！")
        print(f"结果列: {NORMALIZED_CONCLUSION_COL}, {NORMALIZED_IMPRESSION_COL}")
        print(f"输出行数: {len(df_final)}")

    except Exception as e:
        print(f"规范化处理过程中出错: {e}")


if __name__ == "__main__":
    print("解剖部位标注诊断处理器 - 应用规范化版本")
    print("=" * 50)
    main()
