# Input: 原始数据 (df_id_train.csv, df_train.csv, fee_detail.csv)
# Output: 预处理后的合并数据, 缺失值/异常值处理报告
# Pos: 数据预处理模块，为特征工程提供清洗后的数据
# Warning: 更新时同步更新注释和 _ARCH.md

"""
数据预处理模块
- 缺失值处理
- 异常值处理
- 数据类型转换
- 三表合并
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """医保数据预处理器"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.df_id = None
        self.df_train = None
        self.fee_detail = None
        self.report = {}

    def load_data(self) -> None:
        """加载原始数据"""
        print('[1] 加载原始数据...')
        self.df_id = pd.read_csv(self.data_dir / 'df_id_train.csv')
        self.df_train = pd.read_csv(self.data_dir / 'df_train.csv')
        self.fee_detail = pd.read_csv(self.data_dir / 'fee_detail.csv')

        print(f'  df_id: {self.df_id.shape}')
        print(f'  df_train: {self.df_train.shape}')
        print(f'  fee_detail: {self.fee_detail.shape}')

    def analyze_missing(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """分析缺失值"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        result = pd.DataFrame({
            '缺失数': missing,
            '缺失率%': missing_pct
        })
        result = result[result['缺失数'] > 0].sort_values('缺失率%', ascending=False)
        return result

    def handle_missing_values(self) -> Dict:
        """处理缺失值"""
        print('\n[2] 处理缺失值...')

        # 分析缺失值
        missing_train = self.analyze_missing(self.df_train, 'df_train')
        missing_fee = self.analyze_missing(self.fee_detail, 'fee_detail')

        # 高缺失率特征 (>80%) - 删除
        high_missing_cols = missing_train[missing_train['缺失率%'] > 80].index.tolist()
        print(f'  高缺失率(>80%)特征数: {len(high_missing_cols)}')

        # 记录删除的列
        self.report['deleted_cols'] = high_missing_cols

        # 删除高缺失率列
        self.df_train = self.df_train.drop(columns=high_missing_cols, errors='ignore')

        # 中低缺失率特征 - 填充
        # 数值型用中位数，类别型用众数
        remaining_missing = self.analyze_missing(self.df_train, 'df_train')

        filled_cols = []
        for col in remaining_missing.index:
            if col in self.df_train.columns:
                # 尝试转为数值
                numeric_col = pd.to_numeric(self.df_train[col], errors='coerce')
                if numeric_col.notna().sum() > 0:
                    # 数值型：用中位数填充
                    fill_value = numeric_col.median()
                    self.df_train[col] = numeric_col.fillna(fill_value)
                else:
                    # 类别型：用众数填充
                    mode_val = self.df_train[col].mode()
                    if len(mode_val) > 0:
                        self.df_train[col] = self.df_train[col].fillna(mode_val[0])
                filled_cols.append(col)

        print(f'  填充的特征数: {len(filled_cols)}')

        # 处理 fee_detail 缺失值
        fee_missing_cols = missing_fee[missing_fee['缺失率%'] > 80].index.tolist()
        self.fee_detail = self.fee_detail.drop(columns=fee_missing_cols, errors='ignore')

        self.report['missing'] = {
            'train_deleted': high_missing_cols,
            'train_filled': filled_cols,
            'fee_deleted': fee_missing_cols
        }

        return self.report['missing']

    def handle_outliers(self) -> Dict:
        """处理异常值 - 3σ规则"""
        print('\n[3] 处理异常值...')

        # 费用相关列
        fee_cols = [col for col in self.df_train.columns if '金额' in col or '费' in col]

        outlier_stats = {}
        for col in fee_cols:
            if col in self.df_train.columns:
                # 转为数值
                self.df_train[col] = pd.to_numeric(self.df_train[col], errors='coerce')

                data = self.df_train[col].dropna()
                if len(data) > 0:
                    mean = data.mean()
                    std = data.std()

                    # 3σ规则
                    lower = max(0, mean - 3 * std)  # 金额不能为负
                    upper = mean + 3 * std

                    # 统计异常值数量
                    outliers = ((self.df_train[col] < lower) | (self.df_train[col] > upper)).sum()

                    # 截断异常值
                    self.df_train[col] = self.df_train[col].clip(lower=lower, upper=upper)

                    if outliers > 0:
                        outlier_stats[col] = {
                            'count': int(outliers),
                            'lower': round(lower, 2),
                            'upper': round(upper, 2)
                        }

        print(f'  处理的费用列数: {len(fee_cols)}')
        print(f'  有异常值的列数: {len(outlier_stats)}')

        self.report['outliers'] = outlier_stats
        return outlier_stats

    def convert_data_types(self) -> Dict:
        """转换数据类型"""
        print('\n[4] 转换数据类型...')

        # 时间字段
        time_cols = ['交易时间', '住院开始时间', '住院终止时间', '申报受理时间', '操作时间']
        time_converted = []

        for col in time_cols:
            if col in self.df_train.columns:
                self.df_train[col] = pd.to_datetime(self.df_train[col], errors='coerce')
                time_converted.append(col)

        # fee_detail 时间
        if '费用发生时间' in self.fee_detail.columns:
            self.fee_detail['费用发生时间'] = pd.to_datetime(
                self.fee_detail['费用发生时间'], errors='coerce'
            )
            time_converted.append('费用发生时间(fee)')

        # 数值字段转换
        numeric_converted = 0
        for col in self.df_train.columns:
            if '金额' in col or '费' in col or '天数' in col:
                self.df_train[col] = pd.to_numeric(self.df_train[col], errors='coerce')
                numeric_converted += 1

        print(f'  时间字段: {len(time_converted)}')
        print(f'  数值字段: {numeric_converted}')

        self.report['type_conversion'] = {
            'time_cols': time_converted,
            'numeric_count': numeric_converted
        }

        return self.report['type_conversion']

    def merge_tables(self) -> pd.DataFrame:
        """合并三表数据"""
        print('\n[5] 合并数据表...')

        # 1. df_train 与 df_id 合并 (个人编码)
        df_merged = self.df_train.merge(
            self.df_id,
            on='个人编码',
            how='left'
        )
        print(f'  df_train + df_id: {df_merged.shape}')

        # 2. 聚合 fee_detail 到每条就诊记录
        # 按顺序号聚合费用明细
        fee_agg = self.fee_detail.groupby('顺序号').agg({
            '单价': ['sum', 'mean', 'max'],
            '数量': ['sum', 'mean'],
            '三目统计项目': lambda x: x.nunique()
        }).reset_index()

        # 扁平化列名
        fee_agg.columns = ['顺序号', '明细总金额', '明细平均单价', '明细最高单价',
                          '明细总数量', '明细平均数量', '三目项目种类数']

        # 合并
        df_merged = df_merged.merge(fee_agg, on='顺序号', how='left')
        print(f'  + fee_detail聚合: {df_merged.shape}')

        # 填充合并后的缺失值
        for col in ['明细总金额', '明细平均单价', '明细最高单价',
                    '明细总数量', '明细平均数量', '三目项目种类数']:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0)

        self.report['merge'] = {
            'final_shape': df_merged.shape,
            'columns': len(df_merged.columns)
        }

        return df_merged

    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """执行完整预处理流程"""
        print('=' * 50)
        print('数据预处理')
        print('=' * 50)

        self.load_data()
        self.handle_missing_values()
        self.handle_outliers()
        self.convert_data_types()
        df_merged = self.merge_tables()

        print('\n' + '=' * 50)
        print('预处理完成!')
        print(f'  最终数据形状: {df_merged.shape}')
        print('=' * 50)

        return df_merged, self.report


def save_report(report: Dict, output_path: Path) -> None:
    """保存预处理报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('# 数据预处理报告\n\n')

        # 缺失值处理
        f.write('## 1. 缺失值处理\n\n')
        if 'missing' in report:
            f.write(f"### 删除的高缺失率列 ({len(report['missing']['train_deleted'])}个):\n")
            for col in report['missing']['train_deleted']:
                f.write(f"- {col}\n")
            f.write(f"\n### 填充的列 ({len(report['missing']['train_filled'])}个):\n")
            for col in report['missing']['train_filled'][:10]:
                f.write(f"- {col}\n")
            if len(report['missing']['train_filled']) > 10:
                f.write(f"- ... 共{len(report['missing']['train_filled'])}个\n")

        # 异常值处理
        f.write('\n## 2. 异常值处理 (3σ规则)\n\n')
        if 'outliers' in report:
            f.write(f"处理了 {len(report['outliers'])} 个有异常值的列\n\n")
            f.write("| 列名 | 异常值数 | 下界 | 上界 |\n")
            f.write("|------|---------|------|------|\n")
            for col, stats in list(report['outliers'].items())[:10]:
                f.write(f"| {col} | {stats['count']} | {stats['lower']} | {stats['upper']} |\n")

        # 类型转换
        f.write('\n## 3. 数据类型转换\n\n')
        if 'type_conversion' in report:
            f.write(f"- 时间字段: {report['type_conversion']['time_cols']}\n")
            f.write(f"- 数值字段数: {report['type_conversion']['numeric_count']}\n")

        # 合并结果
        f.write('\n## 4. 数据合并\n\n')
        if 'merge' in report:
            f.write(f"- 最终形状: {report['merge']['final_shape']}\n")
            f.write(f"- 列数: {report['merge']['columns']}\n")


if __name__ == '__main__':
    # 测试
    ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT / '原始数据'

    preprocessor = DataPreprocessor(DATA_DIR)
    df, report = preprocessor.run()

    # 保存报告
    save_report(report, ROOT / 'docs' / 'preprocessing_report.md')
    print(f"\n报告已保存: {ROOT / 'docs' / 'preprocessing_report.md'}")
