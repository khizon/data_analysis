from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.compose import ColumnTransformer
from optuna import Trial
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional
from utils.transformations import ConcatText, SplitDateTimeTransformer

from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def instantiate_numerical_simple_imputer(trial : Trial, fill_value : int=-1) -> SimpleImputer:
  strategy = trial.suggest_categorical(
    'numerical_strategy', ['mean', 'median', 'most_frequent', 'constant']
  )
  return SimpleImputer(strategy=strategy, fill_value=fill_value)

def instantiate_categorical_simple_imputer(trial : Trial, fill_value : str='missing') -> SimpleImputer:
  strategy = trial.suggest_categorical(
    'categorical_strategy', ['most_frequent', 'constant']
  )
  return SimpleImputer(strategy=strategy, fill_value=fill_value)

def instantiate_target_encoder(trial : Trial) -> TargetEncoder:
  params = {
    'smoothing': trial.suggest_float('smoothing', 1.0, 100.0),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50)
  }
  return TargetEncoder(**params)

def instantiate_robust_scaler(trial : Trial) -> RobustScaler:
  params = {
    'with_centering': trial.suggest_categorical(
      'with_centering', [True, False]
    ),
    'with_scaling': trial.suggest_categorical(
      'with_scaling', [True, False]
    )
  }
  return RobustScaler(**params)

def instantiate_catboostclf(trial : Trial) -> CatBoostClassifier:
  params = {
    'iterations': trial.suggest_int('cat_iterations', 50, 1000),
    'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.3),
    'depth': trial.suggest_int('cat_depth', 2, 10),
    'l2_leaf_reg': trial.suggest_int('cat_l2_leaf_reg', 2, 30),
    'border_count': trial.suggest_int('cat_border_count', 1, 255),
    'loss_function': 'MultiClass',
    'verbose': False,
    'random_state': 42,
    'task_type': 'GPU',
    'devices' :'0'
  }
  return CatBoostClassifier(**params)

def instantiate_xgbclf(trial : Trial) -> XGBClassifier:
  params = {
    'max_depth': trial.suggest_int('xgb_max_depth', 2, 10),
    'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
    'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 1000),
    'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
    'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
    'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-5, 10.0),
    'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-5, 10.0),
    'device': 'cuda',
    'tree_method': 'gpu_hist',  # this enables GPU.
    'verbosity': 0,
    'seed': 42
  }
  return XGBClassifier(**params)

# def instantiate_lgbmclf(trial : Trial) -> LGBMClassifier:
#   params = {
#     'num_leaves': trial.suggest_int('lgbm_num_leaves', 2, 255),
#     'max_depth': trial.suggest_int('lgbm_max_depth', 2, 10),
#     'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.3),
#     'n_estimators': trial.suggest_int('lgbm_n_estimators', 50, 1000),
#     'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 5, 100),
#     'subsample': trial.suggest_float('lgbm_subsample', 0.5, 1.0),
#     'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.5, 1.0),
#     'reg_alpha': trial.suggest_float('lgbm_reg_alpha', 1e-5, 10.0),
#     'reg_lambda': trial.suggest_float('lgbm_reg_lambda', 1e-5, 10.0),
#     'num_class': trial.suggest_int('lgbm_num_class', 2, 10),
#     'verbose': -1,
#     'random_state': 42
#   }
#   return LGBMClassifier(**params)

def instantiate_dim_reduce(trial : Trial) -> SelectPercentile:
  params = {
    'percentile': trial.suggest_int('percentile', 1, 100)
  }
  return SelectPercentile(**params)

def instantiate_numerical_pipeline(trial : Trial) -> Pipeline:
  pipeline = Pipeline([
    ('imputer', instantiate_numerical_simple_imputer(trial)),
    ('scaler', instantiate_robust_scaler(trial))
  ])
  return pipeline

def instantiate_categorical_pipeline(trial : Trial) -> Pipeline:
  pipeline = Pipeline([
    ('imputer', instantiate_categorical_simple_imputer(trial)),
    ('encoder', instantiate_target_encoder(trial))
  ])
  return pipeline

def instantiate_string_pipeline(trial : Trial) -> Pipeline:
  pipeline = Pipeline([
    ('context', ConcatText()),
    ('tfidf', TfidfVectorizer())
  ])

  return pipeline

def instantiate_datetime_pipeline(trial : Trial) -> Pipeline:
  pipeline = Pipeline([
    ('split_date', SplitDateTimeTransformer()),
    ('imputer', instantiate_categorical_simple_imputer(trial)),
    ('encoder', instantiate_target_encoder(trial))
  ])

  return pipeline

def instantiate_processor(trial: Trial, numerical_columns: Optional[list[str]]=None, categorical_columns: Optional[list[str]]=None, datetime_columns: Optional[list[str]]=None, string_columns: Optional[list[str]]=None) -> ColumnTransformer:
  transformers = []

  if numerical_columns is not None:
    numerical_pipeline = instantiate_numerical_pipeline(trial)
    transformers.append(('numerical_pipeline', numerical_pipeline, numerical_columns))

  if categorical_columns is not None:
    categorical_pipeline = instantiate_categorical_pipeline(trial)
    transformers.append(('categorical_pipeline', categorical_pipeline, categorical_columns))

  if datetime_columns is not None:
    datetime_pipeline = instantiate_datetime_pipeline(trial)
    transformers.append(('datetime_pipeline', datetime_pipeline, datetime_columns))

  if string_columns is not None:
    string_pipeline = instantiate_string_pipeline(trial)
    transformers.append(('string_pipeline', string_pipeline, string_columns))

  return ColumnTransformer(transformers)

def instantiate_model(trial : Trial, numerical_columns : list[str], categorical_columns : list[str], datetime_columns : list[str], string_columns : list[str]) -> Pipeline:
  processor = instantiate_processor(
    trial, numerical_columns, categorical_columns, datetime_columns, string_columns
  )

  clf_name = trial.suggest_categorical(
    'clf', ['XGBoost', 'Catboost']
  )

  if clf_name == 'XGBoost':
    clf = instantiate_xgbclf(trial)
  # elif clf_name == 'LightGBM':
  #   clf = instantiate_lgbmclf(trial)
  elif clf_name == 'Catboost':
    clf = instantiate_catboostclf(trial)

  dim_reduce = instantiate_dim_reduce(trial)

  return Pipeline([
    ('processor', processor),
    ('dim_reduce', dim_reduce),
    ('clf', clf)
  ])


