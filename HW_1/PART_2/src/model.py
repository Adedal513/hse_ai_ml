from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from src.preprocess import TechFieldsTransformer, HierarchicalImputer, FeatureEngineeringGenerator
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

numeric = [
    'year',
    'max_torque_rpm',
    'year_outlier',
    'km_driven_outlier',
    'mileage_outlier',
    'engine_outlier',
    'max_power_outlier',
    'seats_outlier',
    'torque_outlier',
    'max_torque_rpm_outlier',
    'log_km_driven',
    'log_mileage',
    'log_engine',
    'log_max_power',
    'log_torque']

categorical = [
    'fuel',
    'seller_type',
    'transmission',
    'owner',
    'brand',
    'seats'
]

columns_to_fill = {
    'seats': [['brand', 'fuel'], ['brand'], []],
    'mileage': [['brand', 'fuel'], ['brand'], []],
    'engine': [['brand', 'fuel'], ['brand'], []],
    'max_power': [['brand', 'fuel'], ['brand'], []],
    'torque': [['brand', 'fuel'], ['brand'], []],
    'max_torque_rpm': [['brand', 'fuel'], ['fuel'], []]
}

potential_outliers = ['km_driven', 'mileage', 'engine', 'max_power', 'torque', 'year', 'seats', 'torque', 'max_torque_rpm']

categorical_ohe = OneHotEncoder(handle_unknown="ignore")

numeric_poly = Pipeline([
    ("scale", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_poly, numeric),
        ("cat", categorical_ohe, categorical)
    ],
    remainder="drop"
)

pipeline = Pipeline([
    ("convert_tech_fields", TechFieldsTransformer()),
    ("impute", HierarchicalImputer(columns_to_fill)),
    ("add_features", FeatureEngineeringGenerator()),
    ("preprocess", preprocess),
    ("ridge", Ridge())
])