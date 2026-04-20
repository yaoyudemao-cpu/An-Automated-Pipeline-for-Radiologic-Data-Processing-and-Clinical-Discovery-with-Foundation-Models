import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import time
import networkx as nx
import matplotlib

from pipeline_config import RAW_SURGERY_CSV, STEP7_OUTPUT_CSV, STEP8_OUTPUT_DIR
from pipeline_utils import ensure_directory, read_csv_with_fallback, require_columns

warnings.filterwarnings('ignore')
matplotlib.use("Agg")

# 创建一个时间戳用于文件夹命名
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = str(STEP8_OUTPUT_DIR / f'analysis_{timestamp}')
ensure_directory(output_dir)
print(f"输出目录: {output_dir}")

# 统计分析相关
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（用于图表中的中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 读取数据
print("正在读取数据...")
# 读取待分析文件
analysis_file_path = STEP7_OUTPUT_CSV
df_analysis = read_csv_with_fallback(analysis_file_path)
require_columns(df_analysis, ['UID', '检查时间'], analysis_file_path.name)
print(f"待分析文件形状: {df_analysis.shape}")

# 读取手术患者名单文件
surgery_file_path = RAW_SURGERY_CSV
df_surgery = read_csv_with_fallback(surgery_file_path)
if 'UID' not in df_surgery.columns and '患者UID' not in df_surgery.columns:
    raise ValueError(f"{surgery_file_path.name} 缺少 UID 或 患者UID 列")
require_columns(df_surgery, ['是否手术', '手术时间'], surgery_file_path.name)
print(f"手术患者名单文件形状: {df_surgery.shape}")

# 检查手术患者名单文件的列名
print("手术患者名单文件列名:")
print(df_surgery.columns.tolist())

# 检查原始数据信息
print(f"\n原始数据行数: {len(df_analysis)}")
print(f"原始数据患者数(唯一UID): {df_analysis['UID'].nunique()}")

# 特征列列表及翻译字典
feature_translation = {
    '非脊柱位置病变': 'Non-spinal lesion',
    '意义不清': 'Unclear significance',
    '无法分类': 'Unclassifiable',
    '阴性结果': 'Negative result',
    '脊柱占位': 'Spinal space-occupying lesion',
    '移行椎': 'Transitional vertebra',
    '类风湿': 'Rheumatoid',
    '神经根袖囊肿': 'Nerve root sleeve cyst',
    '椎旁软组织病变': 'Paravertebral soft tissue lesion',
    '脊髓损伤或变性': 'Spinal cord injury or degeneration',
    '脊髓空洞或中央管扩张': 'Syringomyelia or central canal dilation',
    '骨质增生融合': 'Hyperostosis fusion',
    '骨质侵蚀': 'Bone erosion',
    '椎管狭窄': 'Spinal stenosis',
    '椎间盘突出': 'Disc herniation',
    '寰枕融合': 'Atlanto-occipital fusion',
    '寰椎后弓发育畸形': 'Posterior arch dysplasia',
    'Chiari畸形': 'Chiari malformation',
    '齿突骨质侵蚀': 'Odontoid bone erosion',
    '齿突骨折': 'Odontoid fracture',
    '齿突不连或齿突小骨': 'Odontoid nonunion or os odontoideum',
    '血管瘤': 'Hemangioma',
    '活动受限': 'Limited mobility',
    '骨质疏松': 'Osteoporosis',
    '项韧带钙化': 'Nuchal ligament calcification',
    '椎间孔狭窄': 'Foraminal stenosis',
    '椎间孔扩大': 'Foraminal enlargement',
    'DISH': 'DISH',
    '骨髓水肿': 'Bone marrow edema',
    '齿突双侧间隙不等宽': 'Asymmetric odontoid spaces',
    '寰枢关节脱位': 'Atlantoaxial dislocation',
    '寰枢椎不稳': 'Atlantoaxial instability',
    '韧带增生/骨化': 'Ligament hyperplasia/ossification',
    '退行性变': 'Degenerative change',
    '脊柱侧弯及旋转': 'Scoliosis and rotation',
    '脊柱后凸': 'Kyphosis',
    '脊柱畸形（除外侧弯、旋转及后凸）': 'Spinal deformity (excluding scoliosis, rotation and kyphosis)',
    '滑膜增生变化': 'Synovial hyperplasia changes',
    '强直性脊柱炎': 'Ankylosing spondylitis',
    '颅底凹陷': 'Basilar invagination',
    '椎体滑脱': 'Spondylolisthesis',
    '椎体或附件骨折': 'Vertebral body or appendage fracture',
    '阻滞椎或分割不全': 'Block vertebra or segmentation defect'
}

# 中文特征列列表
chinese_features = list(feature_translation.keys())

# 检查哪些特征列存在于数据中
existing_features = []
for feature in chinese_features:
    if feature in df_analysis.columns:
        existing_features.append(feature)

print(f"\n找到 {len(existing_features)} 个特征列")

# 数据预处理
print("\n正在进行数据预处理...")

# 转换检查日期为datetime
df_analysis['检查时间'] = pd.to_datetime(df_analysis['检查时间'], errors='coerce')

# 处理特征列
print("处理特征列...")
for feature in existing_features:
    # 将非空值转换为1，空值转换为0
    df_analysis[feature] = df_analysis[feature].apply(
        lambda x: 1 if pd.notna(x) and str(x).strip() != '' and str(x).strip() != '0' else 0
    )

print(f"特征处理完成，示例数据：")
for feature in existing_features[:5]:
    print(f"{feature}: {df_analysis[feature].sum()}")

# 创建批次汇总数据
print("\n正在创建批次汇总数据...")

def create_batch_summary(df):
    """创建批次汇总数据"""
    batch_summary_list = []
    
    # 按患者分组
    for uid, group in df.groupby('UID'):
        # 按检查日期排序
        group_sorted = group.sort_values('检查时间')
        dates = group_sorted['检查时间'].tolist()
        
        # 如果只有一个检查，直接作为一个批次
        if len(dates) == 1:
            batch_data = group_sorted.iloc[0].copy()
            summary_record = {
                'UID': uid,
                'batch_id': 1,
                '检查时间': batch_data['检查时间'],
                'batch_size': 1,
                'is_last_batch': True
            }
            
            # 添加特征
            for feature in existing_features:
                summary_record[feature] = batch_data[feature] if feature in batch_data else 0
            
            batch_summary_list.append(summary_record)
        else:
            # 多个检查，需要进行合并
            batches = []
            current_batch = []
            current_date = None
            
            for i, date in enumerate(dates):
                if current_date is None:
                    current_date = date
                    current_batch.append(i)
                else:
                    # 检查是否在31天内
                    if (date - current_date).days <= 31:
                        current_batch.append(i)
                    else:
                        # 开始新批次
                        batches.append(current_batch)
                        current_batch = [i]
                        current_date = date
            
            if current_batch:
                batches.append(current_batch)
            
            # 为每个批次创建汇总数据
            for batch_idx, batch_indices in enumerate(batches, 1):
                batch_data = group_sorted.iloc[batch_indices].copy()
                
                # 创建批次汇总记录
                summary_record = {
                    'UID': uid,
                    'batch_id': batch_idx,
                    '检查时间': batch_data['检查时间'].min(),
                    'batch_size': len(batch_data),
                    'is_last_batch': (batch_idx == len(batches))
                }
                
                # 对于每个特征，如果批次内任意检查有该特征，则为1
                for feature in existing_features:
                    if feature in batch_data.columns:
                        summary_record[feature] = int(batch_data[feature].max())
                    else:
                        summary_record[feature] = 0
                
                batch_summary_list.append(summary_record)
    
    return pd.DataFrame(batch_summary_list)

# 创建批次汇总
batch_summary = create_batch_summary(df_analysis)

print(f"批次汇总完成:")
print(f"批次数据行数: {len(batch_summary)}")
print(f"患者数量: {batch_summary['UID'].nunique()}")
print(f"最后一批次的数量: {batch_summary['is_last_batch'].sum()}")

# 处理手术信息 - 从手术患者名单文件中读取
print("\n正在处理手术信息...")

# 检查手术患者名单文件的列名
print("手术患者名单文件列名:")
print(df_surgery.columns.tolist())

# 根据您提供的列名，手术患者名单文件应该包含"患者UID"、"是否手术"、"手术时间"
# 重命名列以便合并
surgery_info = df_surgery.copy()

# 检查列名并重命名
if '患者UID' in surgery_info.columns:
    surgery_info = surgery_info.rename(columns={'患者UID': 'UID'})
elif 'UID' in surgery_info.columns:
    pass  # 已经正确命名
else:
    print("警告: 未找到患者UID列，尝试查找其他可能的列名")
    # 尝试查找包含"UID"或"患者"的列
    uid_columns = [col for col in surgery_info.columns if 'UID' in col or '患者' in col]
    if uid_columns:
        surgery_info = surgery_info.rename(columns={uid_columns[0]: 'UID'})
    else:
        print("错误: 无法找到患者ID列")
        # 使用第一列作为UID
        surgery_info = surgery_info.rename(columns={surgery_info.columns[0]: 'UID'})

print(f"处理后的手术信息列名: {surgery_info.columns.tolist()}")

# 处理手术状态
if '是否手术' in surgery_info.columns:
    surgery_info['是否手术'] = surgery_info['是否手术'].apply(
        lambda x: 1 if pd.notna(x) and str(x).strip() != '' and str(x).strip().lower() not in ['0', '否', 'no', 'false'] else 0
    )
else:
    print("警告: 未找到'是否手术'列，所有患者标记为未手术")
    surgery_info['是否手术'] = 0

# 处理手术时间
if '手术时间' in surgery_info.columns:
    surgery_info['手术时间'] = pd.to_datetime(surgery_info['手术时间'], errors='coerce')
else:
    print("警告: 未找到'手术时间'列")
    surgery_info['手术时间'] = pd.NaT

# 只保留必要的列
surgery_info = surgery_info[['UID', '是否手术', '手术时间']].drop_duplicates(subset=['UID'])

print(f"手术信息数据形状: {surgery_info.shape}")
print(f"手术患者数: {surgery_info['是否手术'].sum()}")

# 合并到批次数据
batch_summary = pd.merge(batch_summary, surgery_info, on='UID', how='left')

# 对于未在手术名单中的患者，标记为未手术
batch_summary['是否手术'] = batch_summary['是否手术'].fillna(0).astype(int)
batch_summary['手术时间'] = pd.to_datetime(batch_summary['手术时间'], errors='coerce')

print(f"手术信息合并完成:")
print(f"总患者数: {batch_summary['UID'].nunique()}")
print(f"手术患者数: {batch_summary['是否手术'].sum()}")

# 确保是否手术列存在且为数值
batch_summary['是否手术'] = pd.to_numeric(batch_summary['是否手术'], errors='coerce').fillna(0).astype(int)

# 获取每个患者的最后一批数据
patient_last_batch = batch_summary[batch_summary['is_last_batch'] == True].copy()
print(f"\n最终数据汇总:")
print(f"患者总数: {batch_summary['UID'].nunique()}")
print(f"最后一批次患者数: {len(patient_last_batch)}")
print(f"手术患者数（最后一批）: {patient_last_batch['是否手术'].sum()}")
print(f"手术患者比例: {patient_last_batch['是否手术'].sum() / len(patient_last_batch) * 100:.1f}%")

# 分析1: 特征在手术组和非手术组之间的差异
print("\n正在进行特征差异分析...")

def calculate_feature_differences(df, feature_list, surgery_col='是否手术'):
    """特征差异分析，只使用每个患者的最后一批数据"""
    results = []
    
    # 统计手术组和非手术组的人数
    surgery_count = df[surgery_col].sum()
    non_surgery_count = len(df) - surgery_count
    
    print(f"分析的患者总数（最后一批）: {len(df)}")
    print(f"手术组人数: {surgery_count}, 非手术组人数: {non_surgery_count}")
    
    for feature in feature_list:
        # 检查特征列是否存在
        if feature not in df.columns:
            continue
        
        try:
            # 创建列联表
            contingency_table = pd.crosstab(
                df[feature], 
                df[surgery_col]
            )
            
            # 确保是2x2表
            if contingency_table.shape != (2, 2):
                # 补全缺失的行或列
                full_table = pd.DataFrame(0, index=[0, 1], columns=[0, 1])
                
                for i in contingency_table.index:
                    for j in contingency_table.columns:
                        if i in full_table.index and j in full_table.columns:
                            full_table.loc[i, j] = contingency_table.loc[i, j]
                
                contingency_table = full_table
            
            # 检查是否有足够的计数
            if contingency_table.sum().sum() == 0:
                continue
            
            # 使用Fisher精确检验
            try:
                oddsratio, p_value = fisher_exact(contingency_table)
                test_used = "Fisher"
            except:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                test_used = "Chi-square"
            
            # 计算各组比例
            surgery_group = df[df[surgery_col] == 1]
            non_surgery_group = df[df[surgery_col] == 0]
            
            surgery_rate = surgery_group[feature].mean() if len(surgery_group) > 0 else 0
            non_surgery_rate = non_surgery_group[feature].mean() if len(non_surgery_group) > 0 else 0
            
            # 计算比值比 (Odds Ratio) 和置信区间
            a = float(contingency_table.loc[1, 1])  # 有特征且手术
            b = float(contingency_table.loc[1, 0])  # 有特征但未手术
            c = float(contingency_table.loc[0, 1])  # 无特征但手术
            d = float(contingency_table.loc[0, 0])  # 无特征且未手术
            
            # 计算OR和置信区间
            if b > 0 and c > 0 and d > 0 and a > 0:
                or_value = (a * d) / (b * c)
                se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
                log_or = np.log(or_value)
                or_ci_lower = np.exp(log_or - 1.96 * se_log_or)
                or_ci_upper = np.exp(log_or + 1.96 * se_log_or)
            else:
                # 如果有零单元格，使用校正
                a_corr = a + 0.5
                b_corr = b + 0.5
                c_corr = c + 0.5
                d_corr = d + 0.5
                or_value = (a_corr * d_corr) / (b_corr * c_corr)
                se_log_or = np.sqrt(1/a_corr + 1/b_corr + 1/c_corr + 1/d_corr)
                log_or = np.log(or_value)
                or_ci_lower = np.exp(log_or - 1.96 * se_log_or)
                or_ci_upper = np.exp(log_or + 1.96 * se_log_or)
            
            results.append({
                '特征_中文': feature,
                '特征_英文': feature_translation.get(feature, feature),
                '手术组阳性率(%)': round(surgery_rate * 100, 2),
                '非手术组阳性率(%)': round(non_surgery_rate * 100, 2),
                '率差(%)': round((surgery_rate - non_surgery_rate) * 100, 2),
                '检验方法': test_used,
                'P值': round(p_value, 6),
                'OR值': round(or_value, 4) if not np.isnan(or_value) else np.nan,
                'OR_95%CI_下限': round(or_ci_lower, 4) if not np.isnan(or_ci_lower) else np.nan,
                'OR_95%CI_上限': round(or_ci_upper, 4) if not np.isnan(or_ci_upper) else np.nan,
                '手术组阳性数': int(a),
                '手术组阴性数': int(c),
                '非手术组阳性数': int(b),
                '非手术组阴性数': int(d)
            })
            
            if p_value < 0.05:
                print(f"  显著特征: {feature} ({feature_translation.get(feature, feature)}), P值={p_value:.6f}, OR={or_value:.2f}")
                
        except Exception as e:
            print(f"特征 '{feature}' 分析出错: {str(e)[:100]}...")
    
    return pd.DataFrame(results)

# 执行特征差异分析（只使用最后一批数据）
feature_diff_results = calculate_feature_differences(patient_last_batch, existing_features)

# 保存结果
if not feature_diff_results.empty:
    output_path = os.path.join(output_dir, 'feature_difference_results.csv')
    feature_diff_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n特征差异分析结果已保存到 '{output_path}'")
    
    # 统计显著特征
    sig_features = feature_diff_results[feature_diff_results['P值'] < 0.05]
    print(f"分析的特征总数: {len(feature_diff_results)}")
    print(f"有显著差异的特征数(P<0.05): {len(sig_features)}")
    
    # 显示前10个显著特征
    if len(sig_features) > 0:
        print("\n前10个显著特征:")
        print(sig_features[['特征_中文', 'P值', 'OR值', '手术组阳性率(%)', '非手术组阳性率(%)']].head(10))
else:
    print("特征差异分析结果为空")

# 分析2: COX生存分析 - 计算从发现特征到手术的时间
print("\n正在进行COX生存分析（从发现特征到手术的时间）...")

def prepare_cox_data_from_feature_to_surgery(df, feature_list):
    """
    准备COX生存分析数据，计算从发现特征到手术的时间
    
    对于有手术的患者：
    - 事件 = 1 (手术)
    - 时间 = 从特征首次出现到手术的时间 (月)
    
    对于无手术的患者：
    - 事件 = 0 (截尾)
    - 时间 = 从特征首次出现到最后一次检查的时间 (月)
    
    对于从未出现特征的患者：
    - 事件 = 0 (截尾)
    - 时间 = 总随访时间 (月)
    """
    cox_data_list = []
    
    # 按患者分组处理
    for uid, patient_data in df.groupby('UID'):
        # 按检查日期排序
        patient_sorted = patient_data.sort_values('检查时间')
        
        # 获取手术状态和时间
        has_surgery = patient_sorted['是否手术'].max() == 1
        surgery_time = patient_sorted['手术时间'].iloc[0] if has_surgery and not patient_sorted['手术时间'].isna().all() else None
        
        # 获取检查时间范围
        first_exam_date = patient_sorted['检查时间'].min()
        last_exam_date = patient_sorted['检查时间'].max()
        
        if pd.isna(first_exam_date) or pd.isna(last_exam_date):
            continue
        
        # 计算总随访时间（月）
        total_follow_up = (last_exam_date - first_exam_date).days / 30.0
        
        if total_follow_up <= 0:
            continue
        
        # 对每个特征进行分析
        for feature in feature_list:
            if feature not in patient_sorted.columns:
                continue
            
            # 找到该特征首次出现的批次
            feature_positive_batches = patient_sorted[patient_sorted[feature] == 1]
            
            if not feature_positive_batches.empty:
                # 特征出现过
                first_feature_date = feature_positive_batches['检查时间'].min()
                has_feature = 1
                
                if has_surgery and surgery_time is not None and not pd.isna(surgery_time):
                    # 有手术的患者：计算从特征出现到手术的时间
                    time_to_event = (surgery_time - first_feature_date).days / 30.0
                    if time_to_event > 0:
                        event = 1
                    else:
                        # 如果手术在特征出现之前，跳过这个特征
                        continue
                else:
                    # 无手术的患者：计算从特征出现到最后一次检查的时间（截尾）
                    time_to_event = (last_exam_date - first_feature_date).days / 30.0
                    event = 0
            else:
                # 特征从未出现
                has_feature = 0
                
                if has_surgery and surgery_time is not None and not pd.isna(surgery_time):
                    # 有手术但特征从未出现：计算从首次检查到手术的时间
                    time_to_event = (surgery_time - first_exam_date).days / 30.0
                    if time_to_event > 0:
                        event = 1
                    else:
                        continue
                else:
                    # 无手术且特征从未出现：使用总随访时间
                    time_to_event = total_follow_up
                    event = 0
            
            # 确保时间为正
            if time_to_event <= 0:
                continue
            
            cox_data_list.append({
                'UID': uid,
                '特征_中文': feature,
                '特征_英文': feature_translation.get(feature, feature),
                '时间_月': round(time_to_event, 2),
                '事件': event,
                '有特征': has_feature
            })
    
    if cox_data_list:
        cox_df = pd.DataFrame(cox_data_list)
        print(f"COX数据准备完成:")
        print(f"总记录数: {len(cox_df)}")
        print(f"涉及患者数: {cox_df['UID'].nunique()}")
        print(f"事件数(手术): {cox_df['事件'].sum()}")
        print(f"截尾数: {len(cox_df) - cox_df['事件'].sum()}")
        return cox_df
    else:
        print("警告: 没有足够的COX分析数据")
        return pd.DataFrame()

# 准备COX分析数据
cox_df = prepare_cox_data_from_feature_to_surgery(batch_summary, existing_features)

# 执行COX分析
def perform_cox_analysis_per_feature(cox_data, feature_list):
    """对每个特征单独进行COX分析"""
    results = []
    
    if len(cox_data) == 0:
        print("COX分析数据为空，无法进行分析")
        return pd.DataFrame()
    
    print(f"将分析 {len(feature_list)} 个特征")
    
    # 统计每个特征的数据量
    feature_stats = {}
    for feature in feature_list:
        feature_data = cox_data[cox_data['特征_中文'] == feature]
        if len(feature_data) > 0:
            with_feature = feature_data[feature_data['有特征'] == 1]
            without_feature = feature_data[feature_data['有特征'] == 0]
            
            events_with_feature = with_feature['事件'].sum()
            events_without_feature = without_feature['事件'].sum()
            
            feature_stats[feature] = {
                'total': len(feature_data),
                'with_feature': len(with_feature),
                'without_feature': len(without_feature),
                'events_with_feature': events_with_feature,
                'events_without_feature': events_without_feature
            }
    
    print(f"有效特征数量: {len(feature_stats)}")
    
    # 对每个特征进行COX分析
    for feature, stats in feature_stats.items():
        # 检查是否有足够的样本
        if stats['events_with_feature'] < 5 or stats['with_feature'] < 10:
            continue
        
        # 提取该特征的数据
        feature_data = cox_data[cox_data['特征_中文'] == feature].copy()
        
        # 准备生存分析数据
        survival_data = feature_data[['时间_月', '事件', '有特征']].copy()
        survival_data.columns = ['time', 'event', 'feature_present']
        
        # 确保数据类型正确
        survival_data['time'] = pd.to_numeric(survival_data['time'], errors='coerce')
        survival_data['event'] = pd.to_numeric(survival_data['event'], errors='coerce').astype(int)
        survival_data['feature_present'] = pd.to_numeric(survival_data['feature_present'], errors='coerce').astype(int)
        
        # 移除无效数据
        survival_data = survival_data.dropna()
        
        if len(survival_data) < 10:
            continue
        
        try:
            # 使用CoxPHFitter
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(survival_data, duration_col='time', event_col='event')
            
            # 获取结果
            summary = cph.summary
            
            if 'feature_present' in summary.index:
                hr = summary.loc['feature_present', 'exp(coef)']
                hr_ci_lower = summary.loc['feature_present', 'exp(coef) lower 95%']
                hr_ci_upper = summary.loc['feature_present', 'exp(coef) upper 95%']
                p_value = summary.loc['feature_present', 'p']
                
                # 计算中位生存时间
                try:
                    median_survival = cph.median_survival_time_
                    if isinstance(median_survival, pd.Series):
                        median_survival = median_survival.iloc[0] if len(median_survival) > 0 else np.nan
                except:
                    median_survival = np.nan
                
                # 计算从特征出现到手术的平均时间（仅限有事件的患者）
                with_feature_events = feature_data[(feature_data['有特征'] == 1) & (feature_data['事件'] == 1)]
                if len(with_feature_events) > 0:
                    mean_time_to_surgery = with_feature_events['时间_月'].mean()
                else:
                    mean_time_to_surgery = np.nan
                
                results.append({
                    '特征_中文': feature,
                    '特征_英文': feature_translation.get(feature, feature),
                    'HR': round(hr, 4),
                    'HR_95%CI_下限': round(hr_ci_lower, 4),
                    'HR_95%CI_上限': round(hr_ci_upper, 4),
                    'P值': round(p_value, 6),
                    '中位生存时间_月': round(median_survival, 2) if not np.isnan(median_survival) else 'NA',
                    '特征出现到手术平均时间_月': round(mean_time_to_surgery, 2) if not np.isnan(mean_time_to_surgery) else 'NA',
                    '样本数': len(survival_data),
                    '事件数': stats['events_with_feature'] + stats['events_without_feature'],
                    '有特征样本数': stats['with_feature'],
                    '有特征事件数': stats['events_with_feature']
                })
                
                if p_value < 0.05:
                    print(f"  显著特征: {feature} ({feature_translation.get(feature, feature)}), P值={p_value:.6f}, HR={hr:.4f}")
                    if not np.isnan(mean_time_to_surgery):
                        print(f"    特征出现到手术平均时间: {mean_time_to_surgery:.2f} 月")
        
        except Exception as e:
            # 如果COX分析失败，尝试log-rank检验
            try:
                from lifelines.statistics import logrank_test
                
                group1 = survival_data[survival_data['feature_present'] == 1]
                group0 = survival_data[survival_data['feature_present'] == 0]
                
                if len(group1) > 0 and len(group0) > 0:
                    results_lr = logrank_test(
                        group1['time'], 
                        group0['time'],
                        event_observed_A=group1['event'],
                        event_observed_B=group0['event']
                    )
                    
                    p_value = results_lr.p_value
                    
                    # 计算简单的风险比
                    risk1 = group1['event'].sum() / group1['time'].sum() if group1['time'].sum() > 0 else 0
                    risk0 = group0['event'].sum() / group0['time'].sum() if group0['time'].sum() > 0 else 0
                    hr_simple = risk1 / risk0 if risk0 > 0 else np.nan
                    
                    results.append({
                        '特征_中文': feature,
                        '特征_英文': feature_translation.get(feature, feature),
                        'HR': round(hr_simple, 4) if not np.isnan(hr_simple) else 'NA',
                        'HR_95%CI_下限': 'NA',
                        'HR_95%CI_上限': 'NA',
                        'P值': round(p_value, 6),
                        '中位生存时间_月': 'NA',
                        '特征出现到手术平均时间_月': 'NA',
                        '样本数': len(survival_data),
                        '事件数': stats['events_with_feature'] + stats['events_without_feature'],
                        '有特征样本数': stats['with_feature'],
                        '有特征事件数': stats['events_with_feature'],
                        '备注': '使用log-rank检验'
                    })
                    
                    if p_value < 0.05:
                        print(f"  显著特征(Log-rank): {feature} ({feature_translation.get(feature, feature)}), P值={p_value:.6f}")
            
            except Exception as e2:
                print(f"特征 '{feature}' 分析失败: {str(e2)[:100]}...")
    
    return pd.DataFrame(results)

# 执行COX分析
if len(cox_df) > 0:
    cox_results = perform_cox_analysis_per_feature(cox_df, existing_features)
else:
    cox_results = pd.DataFrame()

# 保存COX分析结果
if not cox_results.empty:
    output_path = os.path.join(output_dir, 'cox_feature_to_surgery.csv')
    cox_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nCOX生存分析结果已保存到 '{output_path}'")
    
    # 统计显著特征
    sig_cox = cox_results[cox_results['P值'] < 0.05]
    print(f"分析的特征数量: {len(cox_results)}")
    print(f"COX分析中显著的特征数(P<0.05): {len(sig_cox)}")
    
    # 显示前10个显著特征及平均时间
    if len(sig_cox) > 0:
        print("\n前10个显著特征（按HR排序）:")
        sig_cox_sorted = sig_cox.sort_values('HR', ascending=False)
        for i, (idx, row) in enumerate(sig_cox_sorted.head(10).iterrows(), 1):
            time_str = f"{row['特征出现到手术平均时间_月']}月" if row['特征出现到手术平均时间_月'] != 'NA' else "NA"
            eng_name = feature_translation.get(row['特征_中文'], row['特征_中文'])
            print(f"  {i}. {row['特征_中文']} ({eng_name}): HR={row['HR']}, P={row['P值']}, 平均时间={time_str}")
else:
    print("COX分析结果为空")

# 可视化 - 所有图表使用英文标签
print("\n生成可视化图表（使用英文标签）...")

# 1. 特征频率条形图（基于最后一批数据）
plt.figure(figsize=(18, 14))
feature_freq = []
for feature in existing_features:
    if feature in patient_last_batch.columns:
        freq = patient_last_batch[feature].mean() * 100
        feature_freq.append((feature, freq))

# 按频率排序
feature_freq.sort(key=lambda x: x[1], reverse=True)

# 创建英文标签
english_labels = [feature_translation.get(f, f) for f, _ in feature_freq]
frequencies = [freq for _, freq in feature_freq]

# 取前30个特征
top_n = min(30, len(english_labels))
english_labels_top = english_labels[:top_n]
frequencies_top = frequencies[:top_n]

y_pos = np.arange(len(english_labels_top))

plt.barh(y_pos, frequencies_top, color='steelblue', edgecolor='black')
plt.yticks(y_pos, english_labels_top, fontsize=10)
plt.xlabel('Frequency (%)', fontsize=12, fontweight='bold')
plt.title('Top 30 Most Common Imaging Findings', fontsize=16, fontweight='bold', pad=20)

# 在条形上添加数值
for i, v in enumerate(frequencies_top):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

plt.tight_layout()
output_path = os.path.join(output_dir, '01_Feature_Frequency_Distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"特征频率条形图已保存到 '{output_path}'")

# 2. 手术组 vs 非手术组特征频率对比图
if not feature_diff_results.empty:
    # 获取前20个最常见的特征
    top_features_by_freq = [f for f, _ in feature_freq[:20]]
    
    # 创建数据
    comparison_data = []
    for feature in top_features_by_freq:
        eng_name = feature_translation.get(feature, feature)
        
        # 手术组频率
        surgery_rate = feature_diff_results[feature_diff_results['特征_中文'] == feature]['手术组阳性率(%)'].values
        surgery_rate = surgery_rate[0] if len(surgery_rate) > 0 else 0
        
        # 非手术组频率
        non_surgery_rate = feature_diff_results[feature_diff_results['特征_中文'] == feature]['非手术组阳性率(%)'].values
        non_surgery_rate = non_surgery_rate[0] if len(non_surgery_rate) > 0 else 0
        
        comparison_data.append({
            'Feature': eng_name,
            'Surgery Group': surgery_rate,
            'Non-Surgery Group': non_surgery_rate
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 绘图
    plt.figure(figsize=(16, 12))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    plt.barh(x - width/2, comparison_df['Surgery Group'], width, label='Surgery Group', color='#FF6B6B', edgecolor='black')
    plt.barh(x + width/2, comparison_df['Non-Surgery Group'], width, label='Non-Surgery Group', color='#4ECDC4', edgecolor='black')
    
    plt.yticks(x, comparison_df['Feature'], fontsize=10)
    plt.xlabel('Frequency (%)', fontsize=12, fontweight='bold')
    plt.title('Comparison of Feature Frequency: Surgery vs Non-Surgery Groups', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '02_Surgery_vs_NonSurgery_Comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"手术组与非手术组特征对比图已保存到 '{output_path}'")

# 3. 特征差异森林图（OR值）
if not feature_diff_results.empty:
    # 筛选有显著差异的特征
    sig_features = feature_diff_results[feature_diff_results['P值'] < 0.05].copy()
    
    if not sig_features.empty:
        # 限制显示数量
        display_features = sig_features.head(25)
        
        plt.figure(figsize=(16, 12))
        
        # 按OR值排序
        display_features = display_features.sort_values('OR值', ascending=True)
        
        # 获取英文标签
        english_labels = [feature_translation.get(f, f) for f in display_features['特征_中文']]
        
        # 创建森林图
        y_positions = range(len(display_features))
        
        # 绘制点
        plt.scatter(display_features['OR值'], y_positions, s=100, color='steelblue', alpha=0.7, edgecolors='black', zorder=3)
        
        # 添加误差线
        for i, (idx, row) in enumerate(display_features.iterrows()):
            if not np.isnan(row['OR_95%CI_下限']) and not np.isnan(row['OR_95%CI_上限']):
                plt.hlines(i, row['OR_95%CI_下限'], row['OR_95%CI_上限'], color='steelblue', linewidth=2, zorder=2)
        
        # 添加参考线
        plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, zorder=1, label='OR=1')
        
        # 设置y轴标签
        plt.yticks(y_positions, english_labels, fontsize=10)
        plt.xticks(fontsize=10)
        
        # 设置标签和标题
        plt.xlabel('Odds Ratio (95% CI)', fontsize=12, fontweight='bold')
        plt.title('Significant Associations between Imaging Findings and Surgery', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=11)
        
        # 使用对数刻度
        plt.xscale('log')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '03_Feature_Association_Forest_Plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征差异森林图已保存到 '{output_path}'")

# 4. COX分析森林图（HR值）
if not cox_results.empty:
    sig_cox = cox_results[cox_results['P值'] < 0.05].copy()
    
    if not sig_cox.empty and 'HR' in sig_cox.columns:
        # 限制显示数量
        display_features = sig_cox.head(25)
        
        plt.figure(figsize=(16, 12))
        
        # 按HR值排序
        display_features = display_features.sort_values('HR', ascending=True)
        
        # 获取英文标签
        english_labels = [feature_translation.get(f, f) for f in display_features['特征_中文']]
        
        # 创建森林图
        y_positions = range(len(display_features))
        
        # 绘制点
        plt.scatter(display_features['HR'], y_positions, s=100, color='darkgreen', alpha=0.7, edgecolors='black', zorder=3)
        
        # 添加误差线
        for i, (idx, row) in enumerate(display_features.iterrows()):
            if row['HR_95%CI_下限'] != 'NA' and row['HR_95%CI_上限'] != 'NA':
                plt.hlines(i, float(row['HR_95%CI_下限']), float(row['HR_95%CI_上限']), color='darkgreen', linewidth=2, zorder=2)
        
        # 添加参考线
        plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, zorder=1, label='HR=1')
        
        # 设置y轴标签
        plt.yticks(y_positions, english_labels, fontsize=10)
        plt.xticks(fontsize=10)
        
        # 设置标签和标题
        plt.xlabel('Hazard Ratio (95% CI)', fontsize=12, fontweight='bold')
        plt.title('COX Analysis: Time from Feature Detection to Surgery', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=11)
        
        # 使用对数刻度
        plt.xscale('log')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '04_COX_Analysis_Forest_Plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"COX分析森林图已保存到 '{output_path}'")

# 5. 特征共现网络图（Top 20特征）
print("\n生成特征共现网络图...")
try:
    # 计算特征共现矩阵
    feature_matrix = patient_last_batch[existing_features]
    
    # 选择前20个最常见的特征
    top_features = [f for f, _ in feature_freq[:20]]
    feature_matrix_top = feature_matrix[top_features]
    
    # 计算相关性矩阵
    corr_matrix = feature_matrix_top.corr()
    
    # 创建网络图
    plt.figure(figsize=(16, 12))
    
    # 创建图形
    G = nx.Graph()
    
    # 添加节点
    for feature in top_features:
        eng_name = feature_translation.get(feature, feature)
        G.add_node(eng_name, size=feature_matrix[feature].mean() * 1000 + 100)
    
    # 添加边（只添加相关性大于0.1的边）
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.1:
                feature1 = top_features[i]
                feature2 = top_features[j]
                eng1 = feature_translation.get(feature1, feature1)
                eng2 = feature_translation.get(feature2, feature2)
                G.add_edge(eng1, eng2, weight=abs(corr) * 5)
    
    # 绘制网络图
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # 节点大小基于特征频率
    node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
    
    # 边宽度基于相关性强度
    edge_widths = [G.edges[edge]['weight'] for edge in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.8, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    plt.title('Feature Co-occurrence Network (Top 20 Features)', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '05_Feature_Cooccurrence_Network.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征共现网络图已保存到 '{output_path}'")
    
except Exception as e:
    print(f"生成特征共现网络图时出错: {str(e)}")

# 6. 从特征到手术的时间分布图
if not cox_results.empty:
    sig_cox_sorted = cox_results[cox_results['P值'] < 0.05].sort_values('HR', ascending=False)
    top_features = sig_cox_sorted.head(10)['特征_中文'].tolist()
    
    if top_features:
        # 提取这些特征的COX数据
        feature_time_data = []
        for feature in top_features:
            feature_data = cox_df[(cox_df['特征_中文'] == feature) & (cox_df['有特征'] == 1) & (cox_df['事件'] == 1)]
            if len(feature_data) > 0:
                eng_name = feature_translation.get(feature, feature)
                for _, row in feature_data.iterrows():
                    feature_time_data.append({
                        'Feature': eng_name,
                        'Time_to_Surgery_Months': row['时间_月']
                    })
        
        if feature_time_data:
            time_df = pd.DataFrame(feature_time_data)
            
            plt.figure(figsize=(14, 10))
            
            # 创建小提琴图
            features_sorted = time_df.groupby('Feature')['Time_to_Surgery_Months'].median().sort_values().index
            time_df['Feature'] = pd.Categorical(time_df['Feature'], categories=features_sorted, ordered=True)
            
            # 创建小提琴图
            violin_parts = plt.violinplot([time_df[time_df['Feature'] == f]['Time_to_Surgery_Months'] 
                                          for f in features_sorted], 
                                          positions=range(len(features_sorted)),
                                          showmeans=True, showmedians=True)
            
            # 设置颜色
            for pc in violin_parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.7)
            
            plt.xticks(range(len(features_sorted)), features_sorted, rotation=45, ha='right', fontsize=10)
            plt.ylabel('Time from Feature Detection to Surgery (Months)', fontsize=12, fontweight='bold')
            plt.title('Distribution of Time from Feature Detection to Surgery', fontsize=16, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3, axis='y')
            
            # 添加样本数量标注
            for i, feature in enumerate(features_sorted):
                n = len(time_df[time_df['Feature'] == feature])
                median = time_df[time_df['Feature'] == feature]['Time_to_Surgery_Months'].median()
                plt.text(i, median + 2, f'n={n}\nmed={median:.1f}', ha='center', fontsize=8)
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, '06_Time_to_Surgery_Distribution.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"特征到手术时间分布图已保存到 '{output_path}'")

# 7. 患者特征数量分布图
plt.figure(figsize=(14, 8))

# 计算每个患者的特征数量
patient_feature_counts = patient_last_batch[existing_features].sum(axis=1)

# 按手术状态分组
surgery_patients = patient_last_batch[patient_last_batch['是否手术'] == 1]
non_surgery_patients = patient_last_batch[patient_last_batch['是否手术'] == 0]

surgery_counts = surgery_patients[existing_features].sum(axis=1)
non_surgery_counts = non_surgery_patients[existing_features].sum(axis=1)

# 创建箱线图
box_data = [non_surgery_counts, surgery_counts]
box = plt.boxplot(box_data, labels=['Non-Surgery', 'Surgery'], patch_artist=True)

# 设置颜色
colors = ['lightgreen', 'lightcoral']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('Number of Imaging Findings per Patient', fontsize=12, fontweight='bold')
plt.title('Distribution of Imaging Findings Count by Surgery Status', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 添加统计信息
for i, data in enumerate(box_data, 1):
    mean_val = np.mean(data)
    median_val = np.median(data)
    plt.text(i, max(data) + 0.5, f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}', 
             ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
output_path = os.path.join(output_dir, '07_Feature_Count_Distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"患者特征数量分布图已保存到 '{output_path}'")

# 8. 热图：特征在手术组和非手术组中的频率
if not feature_diff_results.empty:
    # 获取前30个特征
    top_features_by_freq = [f for f, _ in feature_freq[:30]]
    
    # 创建热图数据
    heatmap_data = []
    for feature in top_features_by_freq:
        eng_name = feature_translation.get(feature, feature)
        
        # 获取手术组和非手术组频率
        row = feature_diff_results[feature_diff_results['特征_中文'] == feature]
        if len(row) > 0:
            surgery_rate = row['手术组阳性率(%)'].values[0]
            non_surgery_rate = row['非手术组阳性率(%)'].values[0]
            
            heatmap_data.append({
                'Feature': eng_name,
                'Surgery Group': surgery_rate,
                'Non-Surgery Group': non_surgery_rate
            })
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df.set_index('Feature', inplace=True)
        
        plt.figure(figsize=(12, 16))
        
        # 创建热图
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Frequency (%)'}, linewidths=0.5)
        
        plt.title('Feature Frequency Heatmap: Surgery vs Non-Surgery Groups', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Group', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, '08_Feature_Frequency_Heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"特征频率热图已保存到 '{output_path}'")

# 9. 生存曲线示例（HR最高的特征）
if not cox_results.empty and 'HR' in cox_results.columns:
    # 找出HR值最高的显著特征
    valid_results = cox_results[cox_results['P值'] < 0.05].copy()
    if not valid_results.empty:
        # 将HR转换为数值
        valid_results['HR_numeric'] = pd.to_numeric(valid_results['HR'], errors='coerce')
        valid_results = valid_results.dropna(subset=['HR_numeric'])
        
        if not valid_results.empty:
            top_feature_row = valid_results.loc[valid_results['HR_numeric'].idxmax()]
            top_feature = top_feature_row['特征_中文']
            top_feature_eng = feature_translation.get(top_feature, top_feature)
            
            # 为该特征绘制生存曲线
            if top_feature in cox_df['特征_中文'].values:
                plt.figure(figsize=(12, 8))
                
                # 准备数据
                survival_df = cox_df[cox_df['特征_中文'] == top_feature][['时间_月', '事件', '有特征']].copy()
                survival_df.columns = ['time', 'event', 'feature_present']
                
                # 使用Kaplan-Meier估计
                kmf = KaplanMeierFitter()
                
                # 按特征分组
                groups = sorted(survival_df['feature_present'].unique())
                
                for group in groups:
                    group_data = survival_df[survival_df['feature_present'] == group]
                    label = f'Feature Present = {int(group)}'
                    
                    kmf.fit(group_data['time'], 
                            group_data['event'], 
                            label=label)
                    
                    kmf.plot_survival_function(ax=plt.gca())
                
                plt.xlabel('Time to Surgery (Months)', fontsize=12, fontweight='bold')
                plt.ylabel('Survival Probability', fontsize=12, fontweight='bold')
                plt.title(f'Kaplan-Meier Survival Curves for {top_feature_eng}', fontsize=16, fontweight='bold')
                plt.legend(title='Feature Status', fontsize=11, title_fontsize=12)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                output_path = os.path.join(output_dir, f'09_Survival_Curve_{top_feature_eng[:20]}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"生存曲线图已保存到 '{output_path}'")

# 10. 综合数据概览图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 10.1 患者检查次数分布
check_counts = batch_summary.groupby('UID').size()
axes[0, 0].hist(check_counts, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Number of Batches per Patient', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Distribution of Batches per Patient', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 10.2 批次大小分布
axes[0, 1].hist(batch_summary['batch_size'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Batch Size (Number of Examinations)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Distribution of Batch Sizes', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 10.3 手术状态分布
surgery_counts_last = patient_last_batch['是否手术'].value_counts()
axes[1, 0].bar(['Non-surgery', 'Surgery'], surgery_counts_last.values, 
               color=['lightgreen', 'lightcoral'], edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Number of Patients', fontsize=11)
axes[1, 0].set_title('Surgery Status Distribution', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 在条形上添加数值
for i, v in enumerate(surgery_counts_last.values):
    axes[1, 0].text(i, v + 5, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

# 10.4 特征阳性率分布
feature_positive_rates = []
for feature in existing_features[:20]:  # 只显示前20个特征
    if feature in patient_last_batch.columns:
        rate = patient_last_batch[feature].mean() * 100
        feature_positive_rates.append(rate)

axes[1, 1].hist(feature_positive_rates, bins=20, color='gold', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Feature Positive Rate (%)', fontsize=11)
axes[1, 1].set_ylabel('Number of Features', fontsize=11)
axes[1, 1].set_title('Distribution of Feature Positive Rates', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Data Overview and Quality Check', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
output_path = os.path.join(output_dir, '10_Data_Overview.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"数据概览图已保存到 '{output_path}'")

# 保存关键数据
output_path = os.path.join(output_dir, 'processed_batch_summary.csv')
batch_summary.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n处理后的批次数据已保存到 '{output_path}'")

if len(cox_df) > 0:
    output_path = os.path.join(output_dir, 'cox_analysis_data.csv')
    cox_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"COX分析数据已保存到 '{output_path}'")

# 保存特征翻译表
translation_df = pd.DataFrame([
    {'中文特征名': ch, '英文特征名': feature_translation.get(ch, ch)}
    for ch in existing_features
])
output_path = os.path.join(output_dir, 'feature_translation_table.csv')
translation_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"特征名翻译表已保存到 '{output_path}'")

# 输出摘要报告
print("\n" + "="*70)
print("分析完成！")
print("="*70)
print(f"输出目录: {output_dir}")
print(f"\n原始数据:")
print(f"  行数: {len(df_analysis)}")
print(f"  患者数: {df_analysis['UID'].nunique()}")
print(f"手术患者名单:")
print(f"  行数: {len(df_surgery)}")
print(f"  手术患者数: {surgery_info['是否手术'].sum()}")
print(f"处理后数据:")
print(f"  批次数据行数: {len(batch_summary)}")
print(f"  患者数: {batch_summary['UID'].nunique()}")
print(f"  最后一批患者数: {len(patient_last_batch)}")
print(f"  手术患者数: {patient_last_batch['是否手术'].sum()}")
print(f"  手术患者比例: {patient_last_batch['是否手术'].sum() / len(patient_last_batch) * 100:.1f}%")
print(f"\n分析结果:")
print(f"  特征数量: {len(existing_features)}")
sig_feature_diff = len(feature_diff_results[feature_diff_results['P值'] < 0.05])
print(f"  特征差异分析显著特征数(P<0.05): {sig_feature_diff}")
sig_cox_analysis = len(cox_results[cox_results['P值'] < 0.05]) if not cox_results.empty else 0
print(f"  COX分析显著特征数(P<0.05): {sig_cox_analysis}")

# 如果有显著COX结果，显示平均时间信息
if not cox_results.empty:
    sig_cox = cox_results[cox_results['P值'] < 0.05]
    if len(sig_cox) > 0:
        # 计算有特征出现到手术时间的平均时间
        valid_times = []
        for _, row in sig_cox.iterrows():
            feature = row['特征_中文']
            feature_data = cox_df[(cox_df['特征_中文'] == feature) & (cox_df['有特征'] == 1) & (cox_df['事件'] == 1)]
            if len(feature_data) > 0:
                valid_times.extend(feature_data['时间_月'].tolist())
        
        if valid_times:
            avg_time = np.mean(valid_times)
            print(f"  特征出现到手术平均时间: {avg_time:.2f} 月")
            print(f"  最短时间: {np.min(valid_times):.2f} 月")
            print(f"  最长时间: {np.max(valid_times):.2f} 月")

print(f"\n生成的文件:")
files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv') or f.endswith('.png')])
for file in files:
    print(f"  - {file}")
print("="*70)
