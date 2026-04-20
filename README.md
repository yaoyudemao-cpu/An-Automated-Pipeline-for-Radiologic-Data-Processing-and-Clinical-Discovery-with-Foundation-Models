# Imaging Pipeline

用于影像报告结构化、解剖部位标注、术语规范化、二次分类映射，以及风险因素和 COX 生存分析的脚本型 Python 项目。

## 项目结构

```text
github_upload/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  └─ raw/
│     ├─ test.csv
│     └─ test_surg.csv
├─ docs/
├─ outputs/
└─ scripts/
   ├─ 01_llm_structuring.py
   ├─ 02_anatomical_site_annotation.py
   ├─ 03_generate_cluster_templates.py
   ├─ 04_llm_cluster_review.py
   ├─ 05_apply_normalization.py
   ├─ 06_second_level_clustering.py
   ├─ 07_map_classification_back.py
   └─ 08_risk_factor_and_cox_analysis.py
```

## 数据说明

- `data/raw/test.csv`
  - 用作整条流水线的初始输入
  - 当前包含 5 列: `UID`, `检查时间`, `检查方法`, `诊断结论`, `诊断印象`
- `data/raw/test_surg.csv`
  - 用作第 8 步手术结局数据
  - 当前包含 3 列: `UID`, `是否手术`, `手术时间`

## 运行顺序

建议在项目根目录执行以下脚本:

```bash
python scripts/pipeline_validation.py
python scripts/01_llm_structuring.py
python scripts/02_anatomical_site_annotation.py
python scripts/03_generate_cluster_templates.py
python scripts/04_llm_cluster_review.py
python scripts/05_apply_normalization.py
python scripts/06_second_level_clustering.py
python scripts/07_map_classification_back.py
python scripts/08_risk_factor_and_cox_analysis.py
```


## 每一步的输入输出

1. `01_llm_structuring.py`
   - 输入: `data/raw/test.csv`
   - 输出: `outputs/step1_structured_reports.csv`

2. `02_anatomical_site_annotation.py`
   - 输入: `outputs/step1_structured_reports.csv`
   - 输出: `outputs/step2_anatomical_site_reports.csv`

3. `03_generate_cluster_templates.py`
   - 输入: `outputs/step2_anatomical_site_reports.csv`
   - 输出: `outputs/step3_cluster_templates/`

4. `04_llm_cluster_review.py`
   - 输入: 第 3 步生成的聚类模板
   - 输出: `outputs/step4_llm_reviews/`

5. `05_apply_normalization.py`
   - 输入: 第 2 步结果 + 第 4 步审核结果或第 3 步模板
   - 输出: `outputs/step5_normalization/normalized_reports.csv`

6. `06_second_level_clustering.py`
   - 输入: `outputs/step5_normalization/normalized_reports.csv`
   - 输出: `outputs/step6_second_level_classification/`

7. `07_map_classification_back.py`
   - 输入: 第 5 步结果 + 第 6 步的两份分类矩阵
   - 输出: `outputs/step7_classification_mapping/reports_with_categories.csv`

8. `08_risk_factor_and_cox_analysis.py`
   - 输入: 第 7 步结果 + `data/raw/test_surg.csv`
   - 输出: `outputs/step8_cox_analysis/analysis_时间戳/`

## 环境要求

- Python 3.10 及以上
- 本地 Ollama 服务
  - 默认接口: `http://127.0.0.1:11434`
  - 第 1、4、6 步依赖该服务
- Python 依赖安装:

```bash
pip install -r requirements.txt
```


## 注意事项

- 当前仓库附带的是测试数据，不是完整生产数据。
- 第 8 步依赖 `lifelines`，如果未安装将无法运行 COX 分析。
- 如果本地 Ollama 服务未启动，则第 1、4、6 步不会成功执行。

## 文档

- 输入校验报告: `docs/reports/pipeline_input_validation.md`
- 运行顺序与改动记录: `docs/reports/pipeline_run_order_and_changes.md`
