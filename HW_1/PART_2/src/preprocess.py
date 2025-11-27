from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re
import numpy as np


class TechFieldsTransformer(BaseEstimator, TransformerMixin):
    """ Реализует приведение технических полей к единому формату. """
    def fit(self, X, y=None):
        X = self.transform(X)
        return self

    def transform(self, X):
        df = X.copy()

        # Извлекаем значение бренда из названия авто
        df['brand'] = df['name'].astype(str).apply(lambda x: x.split(' ')[0])
        df = df.drop('name', axis=1)

        df['mileage'] = (
            df['mileage']
            .astype(str)
            .str.replace(' kmpl', '')
            .str.replace(' km/kg', '')
            .str.strip()
        )
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

        df['engine'] = (
            df['engine']
            .astype(str)
            .str.replace(' CC', '')
            .str.strip()
        )
        df['engine'] = pd.to_numeric(df['engine'], errors='coerce')

        df['max_power'] = (
            df['max_power']
            .astype(str)
            .str.replace(' bhp', '')
            .str.strip()
        )
        df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

        # torque parsing
        df['torque_value'] = np.nan
        df['max_torque_rpm'] = np.nan

        for idx, torque_str in df['torque'].items():
            if pd.isna(torque_str):
                continue

            torque_str = str(torque_str).strip().replace(',', '')

            # numeric torque value
            torque_match = re.search(r'([\d.]+)\s*([a-z]*)\s*(nm|kgm)', torque_str, re.IGNORECASE)
            if torque_match:
                torque_val = float(torque_match.group(1))
                unit = torque_match.group(3).lower()

                if unit == 'kgm':
                    torque_val *= 9.80665

                df.at[idx, 'torque_value'] = torque_val

            # rpm
            rpm_range_match = re.search(r'([\d]+)\s*[-~,]+\s*([\d]+).*?rpm', torque_str, re.IGNORECASE)
            if rpm_range_match:
                rpm1 = float(rpm_range_match.group(1))
                rpm2 = float(rpm_range_match.group(2))
                df.at[idx, 'max_torque_rpm'] = (rpm1 + rpm2) / 2
            else:
                rpm_match = re.search(r'@\s*([\d]+)', torque_str)
                if rpm_match:
                    df.at[idx, 'max_torque_rpm'] = float(rpm_match.group(1))
                else:
                    rpm_match = re.search(r'([\d]+)\s*rpm', torque_str)
                    if rpm_match:
                        df.at[idx, 'max_torque_rpm'] = float(rpm_match.group(1))

        df = df.drop('torque', axis=1)

        df = df.rename(columns={
            'torque_value': 'torque'
        })

        return df

class FeatureEngineeringGenerator(BaseEstimator, TransformerMixin):
  def __init__(self, factor=1.5):
        self.potential_outliers = ['km_driven', 'mileage', 'engine', 'max_power', 'torque', 'year', 'seats', 'torque',  'max_torque_rpm']
        self.factor = factor
        self.bounds_ = {}

  def fit(self, X, y=None):
        for col in self.potential_outliers:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor*iqr, q3 + self.factor*iqr)
        return self

  def transform(self, X):
    df = X.copy()

    # Размечаем выбросы
    for col, (low, high) in self.bounds_.items():
        df[col + '_outlier'] = ((df[col] < low) | (df[col] > high)).astype(int)

    # Применяем логарифм к вещественным признакам
    loggable = ["km_driven", "mileage", "engine", "max_power", "torque"]
    for col in loggable:
      if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col].replace(0, 1e-9))
        df = df.drop(col, axis =1)

    # Добавляем признак мощности на литр
    if all(c in df.columns for c in ["max_power", "engine"]):
      df["power_per_liter"] = df["max_power"] / df["engine"]

    # Добавляем средний пробег в год
    if "km_driven" in df.columns and "year" in df.columns:
      current_year = df["year"].max()+1
      df["km_per_year"] = df["km_driven"] / (current_year - df["year"] + 1)

    if "engine" in df.columns and "mileage" in df.columns:
      df["engine_x_mileage"] = df["engine"] * df["mileage"]

    return df

class HierarchicalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_fill):
        self.columns_to_fill = columns_to_fill
        self.medians_ = {}

    def fit(self, X, y=None):
        self.medians_ = {}
        for col, groups_list in self.columns_to_fill.items():
            self.medians_[col] = []
            for group_cols in groups_list:
                if group_cols:
                    medians = X.groupby(group_cols)[col].median()
                    self.medians_[col].append((group_cols, medians))
                else:
                    median_val = X[col].median()
                    self.medians_[col].append(([], median_val))
        return self

    def transform(self, X):
        X = X.copy()
        for col, groups_medians in self.medians_.items():
            for group_cols, medians in groups_medians:
                if group_cols:
                    # заполняем по группам, используя медианы, сохранённые на тренировке
                    X[col] = X.apply(
                        lambda row: medians.get(tuple(row[c] for c in group_cols), row[col])
                        if pd.isna(row[col]) else row[col],
                        axis=1
                    )
                else:
                    # глобальная медиана
                    X[col] = X[col].fillna(medians)
        return X