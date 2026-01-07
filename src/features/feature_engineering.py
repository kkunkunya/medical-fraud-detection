# Input: 预处理后的数据 (preprocessed_data.pkl)
# Output: 就诊特征DataFrame, 时序特征DataFrame, 筛选后的特征
# Pos: 特征工程模块，为模型训练提供特征数据
# Warning: 更新时同步更新注释和 _ARCH.md

"""
特征工程模块
- 就诊特征构建
- 时序特征构建
- 特征筛选
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """特征工程器"""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 预处理后的数据，包含 class 列
        """
        self.df = df.copy()
        self.label_col = 'class'
        self.id_col = '个人编码'
        self.time_col = '交易时间'

        # 特征存储
        self.clinical_features = None  # 就诊特征
        self.temporal_features = None  # 时序特征
        self.report = {}

    def build_clinical_features(self) -> pd.DataFrame:
        """构建就诊特征（按用户聚合）"""
        print('\n[1] 构建就诊特征...')

        # 费用相关列
        fee_cols = [col for col in self.df.columns
                   if ('金额' in col or '费' in col) and col != self.label_col]

        features = {}

        # 1. 就诊行为特征
        print('  1.1 就诊行为特征...')
        behavior = self.df.groupby(self.id_col).agg({
            '顺序号': 'count',  # 就诊次数
            '医院编码': 'nunique',  # 就诊医院数
        }).rename(columns={
            '顺序号': 'visit_count',
            '医院编码': 'hospital_count'
        })

        # 计算就诊频率（如果有时间信息）
        if self.time_col in self.df.columns:
            time_range = self.df.groupby(self.id_col)[self.time_col].agg(['min', 'max'])
            time_range['days'] = (time_range['max'] - time_range['min']).dt.days + 1
            time_range['visit_frequency'] = behavior['visit_count'] / time_range['days'].clip(lower=1)
            behavior['visit_frequency'] = time_range['visit_frequency']

        features['behavior'] = behavior

        # 2. 费用聚合特征
        print('  1.2 费用聚合特征...')
        fee_agg = self.df.groupby(self.id_col)[fee_cols].agg(['sum', 'mean', 'max', 'std'])
        fee_agg.columns = ['_'.join(col).strip() for col in fee_agg.columns.values]
        # 填充NaN（std可能为NaN如果只有一条记录）
        fee_agg = fee_agg.fillna(0)
        features['fee'] = fee_agg

        # 3. 费用比例特征
        print('  1.3 费用比例特征...')
        total_fee = self.df.groupby(self.id_col)[fee_cols].sum().sum(axis=1).replace(0, 1)
        fee_ratio = self.df.groupby(self.id_col)[fee_cols].sum().div(total_fee, axis=0)
        fee_ratio.columns = [f'{col}_ratio' for col in fee_ratio.columns]
        features['fee_ratio'] = fee_ratio

        # 4. 明细特征（如果存在）
        detail_cols = [col for col in self.df.columns if '明细' in col]
        if detail_cols:
            print('  1.4 明细特征...')
            detail_agg = self.df.groupby(self.id_col)[detail_cols].agg(['sum', 'mean'])
            detail_agg.columns = ['_'.join(col).strip() for col in detail_agg.columns.values]
            features['detail'] = detail_agg

        # 合并所有特征
        clinical_df = pd.concat(features.values(), axis=1)

        # 添加标签
        labels = self.df.groupby(self.id_col)[self.label_col].first()
        clinical_df[self.label_col] = labels

        self.clinical_features = clinical_df
        print(f'  就诊特征数: {len(clinical_df.columns) - 1}')
        print(f'  样本数: {len(clinical_df)}')

        return clinical_df

    def build_temporal_features(self) -> pd.DataFrame:
        """构建时序特征（按用户聚合）"""
        print('\n[2] 构建时序特征...')

        if self.time_col not in self.df.columns:
            print('  警告: 无时间列，跳过时序特征')
            return None

        # 确保时间列是datetime类型
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], errors='coerce')

        features = {}

        # 1. 周聚合特征
        print('  2.1 周聚合特征...')
        self.df['week'] = self.df[self.time_col].dt.isocalendar().week
        weekly = self.df.groupby([self.id_col, 'week']).size().unstack(fill_value=0)
        weekly_agg = pd.DataFrame({
            'weekly_visit_mean': weekly.mean(axis=1),
            'weekly_visit_std': weekly.std(axis=1).fillna(0),
            'weekly_visit_max': weekly.max(axis=1),
            'active_weeks': (weekly > 0).sum(axis=1)
        })
        features['weekly'] = weekly_agg

        # 2. 月聚合特征
        print('  2.2 月聚合特征...')
        self.df['month'] = self.df[self.time_col].dt.month
        monthly = self.df.groupby([self.id_col, 'month']).size().unstack(fill_value=0)
        monthly_agg = pd.DataFrame({
            'monthly_visit_mean': monthly.mean(axis=1),
            'monthly_visit_std': monthly.std(axis=1).fillna(0),
            'monthly_visit_max': monthly.max(axis=1),
            'active_months': (monthly > 0).sum(axis=1)
        })
        features['monthly'] = monthly_agg

        # 3. 时间间隔特征
        print('  2.3 时间间隔特征...')
        self.df = self.df.sort_values([self.id_col, self.time_col])
        self.df['time_diff'] = self.df.groupby(self.id_col)[self.time_col].diff().dt.days

        interval_agg = self.df.groupby(self.id_col)['time_diff'].agg([
            ('interval_mean', 'mean'),
            ('interval_std', 'std'),
            ('interval_min', 'min'),
            ('interval_max', 'max')
        ]).fillna(0)

        # 短间隔次数（≤3天）
        interval_agg['short_interval_count'] = self.df[self.df['time_diff'] <= 3].groupby(self.id_col).size()
        interval_agg['short_interval_count'] = interval_agg['short_interval_count'].fillna(0)

        # 同日就诊次数
        interval_agg['same_day_count'] = self.df[self.df['time_diff'] == 0].groupby(self.id_col).size()
        interval_agg['same_day_count'] = interval_agg['same_day_count'].fillna(0)

        features['interval'] = interval_agg

        # 4. 费用趋势特征
        print('  2.4 费用趋势特征...')
        fee_col = '药品费发生金额' if '药品费发生金额' in self.df.columns else None
        if fee_col:
            # 按时间排序后的费用序列
            def calc_trend(group):
                if len(group) < 2:
                    return pd.Series({'fee_trend': 0, 'fee_late_early_ratio': 1})

                values = group[fee_col].values
                x = np.arange(len(values))

                # 线性趋势
                if len(values) > 1 and np.std(values) > 0:
                    slope, _, _, _, _ = stats.linregress(x, values)
                else:
                    slope = 0

                # 晚期/早期费用比率
                mid = len(values) // 2
                early_mean = values[:mid].mean() if mid > 0 else 1
                late_mean = values[mid:].mean() if mid < len(values) else 1
                ratio = late_mean / max(early_mean, 0.01)

                return pd.Series({'fee_trend': slope, 'fee_late_early_ratio': ratio})

            trend_features = self.df.groupby(self.id_col).apply(calc_trend)
            features['trend'] = trend_features

        # 5. 波动特征
        print('  2.5 波动特征...')
        if fee_col:
            fee_stats = self.df.groupby(self.id_col)[fee_col].agg([
                ('fee_cv', lambda x: x.std() / max(x.mean(), 0.01)),  # 变异系数
                ('fee_range', lambda x: x.max() - x.min())
            ])

            # 异常费用次数（超过均值+2*std）
            user_stats = self.df.groupby(self.id_col)[fee_col].agg(['mean', 'std'])
            self.df = self.df.merge(user_stats, on=self.id_col, suffixes=('', '_user'))
            self.df['is_high_fee'] = self.df[fee_col] > (self.df['mean'] + 2 * self.df['std'])
            fee_stats['high_fee_count'] = self.df.groupby(self.id_col)['is_high_fee'].sum()

            features['volatility'] = fee_stats

        # 6. 周期编码特征
        print('  2.6 周期编码特征...')
        # 主要就诊月份的sin/cos编码
        main_month = self.df.groupby(self.id_col)['month'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 6)
        cycle_features = pd.DataFrame({
            'month_sin': np.sin(2 * np.pi * main_month / 12),
            'month_cos': np.cos(2 * np.pi * main_month / 12)
        }, index=main_month.index)

        # 主要就诊星期
        self.df['weekday'] = self.df[self.time_col].dt.weekday
        main_weekday = self.df.groupby(self.id_col)['weekday'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 3)
        cycle_features['weekday_sin'] = np.sin(2 * np.pi * main_weekday / 7)
        cycle_features['weekday_cos'] = np.cos(2 * np.pi * main_weekday / 7)

        # 周末就诊比例
        self.df['is_weekend'] = self.df['weekday'].isin([5, 6])
        cycle_features['weekend_ratio'] = self.df.groupby(self.id_col)['is_weekend'].mean()

        features['cycle'] = cycle_features

        # 合并所有时序特征
        temporal_df = pd.concat(features.values(), axis=1)

        # 添加标签
        labels = self.df.groupby(self.id_col)[self.label_col].first()
        temporal_df[self.label_col] = labels

        # 填充NaN
        temporal_df = temporal_df.fillna(0)

        self.temporal_features = temporal_df
        print(f'  时序特征数: {len(temporal_df.columns) - 1}')
        print(f'  样本数: {len(temporal_df)}')

        return temporal_df

    def filter_by_variance(self, df: pd.DataFrame, threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """方差阈值筛选"""
        print('\n[3] 方差阈值筛选...')

        feature_cols = [col for col in df.columns if col != self.label_col]
        X = df[feature_cols].values

        # 标准化后计算方差
        X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_normalized)

        selected_mask = selector.get_support()
        selected_cols = [col for col, selected in zip(feature_cols, selected_mask) if selected]
        removed_cols = [col for col, selected in zip(feature_cols, selected_mask) if not selected]

        print(f'  原始特征数: {len(feature_cols)}')
        print(f'  保留特征数: {len(selected_cols)}')
        print(f'  移除特征数: {len(removed_cols)}')

        result_df = df[selected_cols + [self.label_col]]
        return result_df, removed_cols

    def filter_by_correlation(self, df: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """相关性筛选"""
        print('\n[4] 相关性筛选...')

        feature_cols = [col for col in df.columns if col != self.label_col]
        corr_matrix = df[feature_cols].corr().abs()

        # 找出高相关的特征对
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print(f'  原始特征数: {len(feature_cols)}')
        print(f'  高相关特征数: {len(to_drop)}')
        print(f'  保留特征数: {len(feature_cols) - len(to_drop)}')

        result_df = df.drop(columns=to_drop)
        return result_df, corr_matrix

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """执行完整特征工程流程"""
        print('=' * 50)
        print('特征工程')
        print('=' * 50)

        # 构建特征
        clinical_df = self.build_clinical_features()
        temporal_df = self.build_temporal_features()

        # 合并特征
        print('\n[5] 合并特征...')
        if temporal_df is not None:
            # 确保索引对齐
            combined_df = clinical_df.drop(columns=[self.label_col]).join(
                temporal_df.drop(columns=[self.label_col]), how='inner'
            )
            combined_df[self.label_col] = clinical_df.loc[combined_df.index, self.label_col]
        else:
            combined_df = clinical_df

        print(f'  合并后特征数: {len(combined_df.columns) - 1}')

        # 特征筛选
        filtered_df, removed_variance = self.filter_by_variance(combined_df)
        final_df, corr_matrix = self.filter_by_correlation(filtered_df)

        print('\n' + '=' * 50)
        print('特征工程完成!')
        print(f'  最终特征数: {len(final_df.columns) - 1}')
        print(f'  样本数: {len(final_df)}')
        print('=' * 50)

        self.report = {
            'clinical_features': len(clinical_df.columns) - 1,
            'temporal_features': len(temporal_df.columns) - 1 if temporal_df is not None else 0,
            'combined_features': len(combined_df.columns) - 1,
            'final_features': len(final_df.columns) - 1,
            'removed_by_variance': removed_variance,
            'correlation_matrix': corr_matrix
        }

        return clinical_df, temporal_df, final_df, self.report


if __name__ == '__main__':
    # 测试
    ROOT = Path(__file__).resolve().parent.parent.parent
    df = pd.read_pickle(ROOT / 'outputs' / 'preprocessed_data.pkl')

    engineer = FeatureEngineer(df)
    clinical_df, temporal_df, final_df, report = engineer.run()

    # 保存
    clinical_df.to_pickle(ROOT / 'outputs' / 'clinical_features.pkl')
    temporal_df.to_pickle(ROOT / 'outputs' / 'temporal_features.pkl')
    final_df.to_pickle(ROOT / 'outputs' / 'final_features.pkl')

    print(f"\n特征已保存到 outputs/")
