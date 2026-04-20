# diagnosis_processor_generate.py - 从解剖部位标注结果生成聚类 JSON 和 CSV 审核模板

import json

import pandas as pd

from pipeline_config import (
    ANATOMICAL_SITE_CONCLUSION_COL,
    ANATOMICAL_SITE_IMPRESSION_COL,
    STEP2_OUTPUT_CSV,
    STEP3_CONCLUSION_JSON,
    STEP3_CONCLUSION_TEMPLATE_CSV,
    STEP3_IMPRESSION_JSON,
    STEP3_IMPRESSION_TEMPLATE_CSV,
)
from pipeline_utils import ensure_directory, read_csv_with_fallback, require_columns


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

        print(
            f"共提取到 {len(all_items)} 个诊断项，"
            f"{len(diagnosis_counts)} 个唯一{count_col_name}"
        )

        return df_items, diagnosis_counts

    def cluster_diagnoses(self, diagnoses, frequency_dict, similarity_threshold=0.6):
        """使用编辑距离和 Jaccard 相似度进行聚类"""
        print("使用相似度计算进行诊断项聚类...")

        clusters = []
        diagnosis_to_cluster = {}

        sorted_diagnoses = sorted(
            diagnoses,
            key=lambda x: frequency_dict.get(x, 0),
            reverse=True,
        )

        for diagnosis in sorted_diagnoses:
            if diagnosis in diagnosis_to_cluster:
                continue

            current_cluster = [diagnosis]
            diagnosis_to_cluster[diagnosis] = len(clusters)

            for other_diagnosis in sorted_diagnoses:
                if other_diagnosis in diagnosis_to_cluster:
                    continue

                similarity = self.calculate_similarity(diagnosis, other_diagnosis)
                if similarity >= similarity_threshold:
                    current_cluster.append(other_diagnosis)
                    diagnosis_to_cluster[other_diagnosis] = len(clusters)

            clusters.append(current_cluster)

        print(f"生成 {len(clusters)} 个聚类")
        return clusters

    def calculate_similarity(self, text1, text2):
        """计算文本相似度（保持原有的 Jaccard + 编辑距离方法）"""
        if not text1 or not text2:
            return 0.0

        set1 = set(text1)
        set2 = set(text2)

        if not set1 or not set2:
            return 0.0

        intersection = set1.intersection(set2)
        union = set1.union(set2)
        jaccard_similarity = len(intersection) / len(union) if union else 0.0

        def normalized_edit_distance(s1, s2):
            if not s1 and not s2:
                return 0.0
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                return 0.0
            distance = self.simple_edit_distance(s1, s2)
            return 1.0 - (distance / max_len)

        edit_similarity = normalized_edit_distance(text1, text2)
        final_similarity = 0.6 * jaccard_similarity + 0.4 * edit_similarity
        return final_similarity

    def simple_edit_distance(self, s1, s2):
        """简化版编辑距离计算"""
        if len(s1) < len(s2):
            return self.simple_edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def generate_cluster_files(self, diagnosis_counts, json_file, csv_file, count_col_name):
        """生成聚类 JSON 文件和 CSV 审核模板"""
        ensure_directory(json_file.parent)
        ensure_directory(csv_file.parent)

        all_diagnoses = diagnosis_counts[count_col_name].tolist()
        frequency_dict = diagnosis_counts.set_index(count_col_name)["出现次数"].to_dict()

        clusters = self.cluster_diagnoses(all_diagnoses, frequency_dict)

        cluster_json = {}
        for i, cluster in enumerate(clusters, 1):
            cluster_json[str(i)] = cluster

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(cluster_json, f, ensure_ascii=False, indent=2)

        print(f"聚类 JSON 文件已生成: {json_file}")

        csv_data = []
        for i, cluster in enumerate(clusters, 1):
            cluster_with_freq = [(diagnosis, frequency_dict.get(diagnosis, 0)) for diagnosis in cluster]
            cluster_with_freq.sort(key=lambda x: x[1], reverse=True)
            representative = cluster_with_freq[0][0]

            diagnosis_with_freq = [f"{diagnosis}({frequency_dict.get(diagnosis, 0)})" for diagnosis in cluster]
            diagnosis_str = "; ".join(diagnosis_with_freq)

            csv_data.append(
                {
                    "聚类ID": i,
                    "聚类大小": len(cluster),
                    "总出现次数": sum(frequency_dict.get(diagnosis, 0) for diagnosis in cluster),
                    "聚类代表": representative,
                    "聚类内诊断结论": diagnosis_str,
                    "推荐规范名称": representative,
                    "最终规范名称": "",
                    "审核结果": "",
                    "备注": "",
                }
            )

        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_file, index=False, encoding="utf-8-sig")

        print(f"CSV 审核模板已生成: {csv_file}")
        return json_file, csv_file


def main():
    """生成聚类 JSON 文件和 CSV 审核模板"""
    input_file = STEP2_OUTPUT_CSV

    if not input_file.exists():
        print(f"输入文件不存在: {input_file}")
        return

    processor = SegmentDiagnosisProcessor()

    try:
        print("步骤1: 读取 CSV 数据文件...")
        data = processor.read_file(input_file)
        require_columns(
            data,
            [ANATOMICAL_SITE_CONCLUSION_COL, ANATOMICAL_SITE_IMPRESSION_COL],
            input_file.name,
        )

        task_settings = [
            (
                ANATOMICAL_SITE_CONCLUSION_COL,
                "诊断结论",
                STEP3_CONCLUSION_JSON,
                STEP3_CONCLUSION_TEMPLATE_CSV,
            ),
            (
                ANATOMICAL_SITE_IMPRESSION_COL,
                "诊断印象",
                STEP3_IMPRESSION_JSON,
                STEP3_IMPRESSION_TEMPLATE_CSV,
            ),
        ]

        for source_col, count_col_name, json_file, csv_file in task_settings:
            print(f"\n步骤2: 提取 {count_col_name} 项...")
            _, diagnosis_counts = processor.extract_segment_diagnosis_items(source_col, count_col_name)

            print(f"\n步骤3: 生成 {count_col_name} 聚类文件...")
            processor.generate_cluster_files(diagnosis_counts, json_file, csv_file, count_col_name)

        print("\n聚类模板文件已生成，请按以下步骤操作:")
        print(f"1. 查看 {STEP3_CONCLUSION_JSON.name} 和 {STEP3_IMPRESSION_JSON.name} 了解聚类结构")
        print(f"2. 打开 {STEP3_CONCLUSION_TEMPLATE_CSV.name} 和 {STEP3_IMPRESSION_TEMPLATE_CSV.name} 进行审核")
        print("3. 在 CSV 模板中填写“最终规范名称”“审核结果”“备注”列，或直接运行第 4 步脚本进行 LLM 审核")
        print(f"4. 当前输入数据总行数: {len(data)}")

    except Exception as e:
        print(f"处理过程中出错: {e}")


if __name__ == "__main__":
    print("解剖部位标注诊断处理器 - 生成聚类模板版本")
    print("=" * 50)
    main()
