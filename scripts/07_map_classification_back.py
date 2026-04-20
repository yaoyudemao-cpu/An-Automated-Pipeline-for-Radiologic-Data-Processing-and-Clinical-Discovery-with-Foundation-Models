import pandas as pd

from pipeline_config import (
    NORMALIZED_CONCLUSION_COL,
    NORMALIZED_IMPRESSION_COL,
    STEP5_OUTPUT_CSV,
    STEP6_CONCLUSION_FINAL_CSV,
    STEP6_IMPRESSION_FINAL_CSV,
    STEP7_OUTPUT_CSV,
)
from pipeline_utils import ensure_directory, read_csv_with_fallback, require_columns


class FixedClassificationMapperV2:
    def __init__(self, original_file_path, classified_file_paths):
        """初始化映射器"""
        self.original_file_path = original_file_path
        self.classified_file_paths = classified_file_paths

        print(f"正在读取原始数据: {original_file_path}")
        self.original_data = read_csv_with_fallback(original_file_path)

        self.classification_columns = set()
        self.classified_tables = []

        for classified_file_path in classified_file_paths:
            print(f"正在读取分类文件: {classified_file_path}")
            classified_data = read_csv_with_fallback(classified_file_path)
            require_columns(classified_data, ["诊断名词"], classified_file_path.name)

            classification_columns = [col for col in classified_data.columns if col != "诊断名词"]
            self.classification_columns.update(classification_columns)
            self.classified_tables.append((classified_file_path, classified_data, classification_columns))

        self.classification_columns = sorted(self.classification_columns)
        print(f"识别到 {len(self.classification_columns)} 个分类列")

    def create_diagnosis_mapping(self):
        """创建诊断名词到分类的映射"""
        print("正在创建诊断名词到分类的映射...")
        diagnosis_to_categories = {}

        for file_path, classified_data, classification_columns in self.classified_tables:
            print(f"处理分类文件: {file_path.name}")

            for _, row in classified_data.iterrows():
                diagnosis = str(row.get("诊断名词", "")).strip()
                if not diagnosis:
                    continue

                categories = diagnosis_to_categories.setdefault(diagnosis, set())
                for col in classification_columns:
                    cell_value = row.get(col)
                    if pd.isna(cell_value):
                        continue

                    cell_str = str(cell_value).strip()
                    if cell_str and cell_str not in ["0", "0.0", "", "nan", "NaN", "None"]:
                        categories.add(col)

        diagnosis_to_categories = {
            diagnosis: sorted(categories)
            for diagnosis, categories in diagnosis_to_categories.items()
            if categories
        }

        print(f"成功为 {len(diagnosis_to_categories)} 个诊断名词创建分类映射")
        for index, (diagnosis, categories) in enumerate(list(diagnosis_to_categories.items())[:5], start=1):
            print(f"  {index}. {diagnosis} -> {', '.join(categories)}")

        return diagnosis_to_categories

    def map_to_original(self, diagnosis_to_categories):
        """将分类映射到原始数据"""
        print("\n正在将分类映射到原始数据...")

        for col in self.classification_columns:
            if col not in self.original_data.columns:
                self.original_data[col] = ""

        processed_count = 0
        classification_count = 0
        source_columns = [NORMALIZED_CONCLUSION_COL, NORMALIZED_IMPRESSION_COL]

        for idx, row in self.original_data.iterrows():
            category_content = {col: [] for col in self.classification_columns}
            has_any_source_text = False

            for source_col in source_columns:
                diagnosis_text = row.get(source_col, "")
                if pd.isna(diagnosis_text) or not str(diagnosis_text).strip():
                    continue

                has_any_source_text = True
                items = str(diagnosis_text).split("；")

                for item in items:
                    item = item.strip()
                    if not item:
                        continue

                    if "@" in item:
                        segment, diagnosis = item.split("@", 1)
                        segment = segment.strip()
                        diagnosis = diagnosis.strip()
                        full_item = f"{segment}@{diagnosis}"
                    else:
                        diagnosis = item.strip()
                        full_item = diagnosis

                    if diagnosis and diagnosis in diagnosis_to_categories:
                        for category in diagnosis_to_categories[diagnosis]:
                            category_content[category].append(full_item)
                            classification_count += 1

            if not has_any_source_text:
                continue

            for col in self.classification_columns:
                self.original_data.at[idx, col] = "；".join(category_content[col]) if category_content[col] else ""

            processed_count += 1

        print(f"已处理 {processed_count} 行原始数据")
        print(f"共映射 {classification_count} 个分类条目")
        return self.original_data

    def save_result(self, output_path):
        """保存结果"""
        ensure_directory(output_path.parent)
        self.original_data.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"结果已保存到: {output_path}")
        return output_path

    def generate_statistics(self):
        """生成统计信息"""
        print("\n" + "=" * 60)
        print("分类统计")
        print("=" * 60)

        total_rows = len(self.original_data)
        rows_with_classification = 0
        category_stats = []

        for col in self.classification_columns:
            non_empty_rows = self.original_data[col].apply(lambda x: str(x).strip() != "").sum()

            item_count = 0
            for content in self.original_data[col]:
                if pd.isna(content) or str(content).strip() == "":
                    continue
                items = str(content).split("；")
                item_count += len(items)

            category_stats.append({"分类": col, "行数": non_empty_rows, "条目数": item_count})

        for _, row in self.original_data.iterrows():
            has_content = any(str(row.get(col, "")).strip() for col in self.classification_columns)
            if has_content:
                rows_with_classification += 1

        print(f"总行数: {total_rows}")
        print(f"有分类内容的行数: {rows_with_classification}")
        print(f"覆盖率: {rows_with_classification / total_rows * 100:.1f}%" if total_rows > 0 else "0.0%")

        category_stats_sorted = sorted(category_stats, key=lambda x: x["行数"], reverse=True)
        print("\n各分类分布（按行数排序）:")
        print("-" * 60)
        for stat in category_stats_sorted:
            if stat["行数"] > 0:
                print(f"  {stat['分类']:<15}: {stat['行数']:>4} 行, {stat['条目数']:>4} 个条目")


def main():
    """主函数"""
    original_file = STEP5_OUTPUT_CSV
    classified_files = [STEP6_CONCLUSION_FINAL_CSV, STEP6_IMPRESSION_FINAL_CSV]

    if not original_file.exists():
        print(f"原始文件不存在: {original_file}")
        return

    missing_files = [path for path in classified_files if not path.exists()]
    if missing_files:
        for path in missing_files:
            print(f"分类文件不存在: {path}")
        return

    mapper = FixedClassificationMapperV2(original_file, classified_files)
    require_columns(
        mapper.original_data,
        [NORMALIZED_CONCLUSION_COL, NORMALIZED_IMPRESSION_COL],
        original_file.name,
    )

    diagnosis_mapping = mapper.create_diagnosis_mapping()
    if not diagnosis_mapping:
        print("错误: 没有找到有效的分类映射")
        return

    mapper.map_to_original(diagnosis_mapping)
    mapper.save_result(STEP7_OUTPUT_CSV)
    mapper.generate_statistics()

    print("\n映射完成!")
    print(f"结果文件: {STEP7_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
