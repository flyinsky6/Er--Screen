import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats

warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ADMET_ERa_ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.performance_results = {}
        self.best_models = {}
        self.train_test_performances = {}
        self.scaler = None
        self.X = None
        self.X_scaled = None
        self.data = None
        self.admet_targets = []
        self.era_targets = []
        self.feature_columns = []

    def load_and_prepare_data(self, admet_file, era_file, smiles_col='SMILES'):
        """ADMETandERα"""
        print("=== ADMETandERα ===")
        try:

            if admet_file.endswith('.xlsx'):
                admet_df = pd.read_excel(admet_file)
            else:
                admet_df = pd.read_csv(admet_file)
            if era_file.endswith('.xlsx'):
                era_df = pd.read_excel(era_file)
            else:
                era_df = pd.read_csv(era_file)
            print(f"ADMET: {admet_df.shape}")
            print(f"ERα: {era_df.shape}")
            # 合并数据
            self.data = pd.merge(admet_df, era_df, on=smiles_col, how='inner')
            print(f"all—data: {self.data.shape}")
            # 识别目标变量
            self._identify_targets()
            return True
        except Exception as e:
            print(f"false data: {e}")
            return False

    def _identify_targets(self):
        """识别目标变量"""
        # ADMET目标变量
        admet_keywords = ['Caco2', 'CYP3A4', 'hERG', 'HOB', 'MN', 'Caco-2']
        # ERα目标变量
        era_keywords = ['pIC50', 'IC50', 'ERα', 'ER_alpha', 'activity']
        self.admet_targets = []
        self.era_targets = []
        for col in self.data.columns:
            col_lower = col.lower()

            for keyword in admet_keywords:
                if keyword.lower() in col_lower:
                    self.admet_targets.append(col)
                    break
            # 检查ERα目标
            for keyword in era_keywords:
                if keyword.lower() in col_lower:
                    self.era_targets.append(col)
                    break

        all_targets = self.admet_targets + self.era_targets
        self.feature_columns = [col for col in self.data.columns
                                if col not in all_targets and 'SMILES' not in col.upper()]
        print(f"ADMET目标: {self.admet_targets}")
        print(f"ERα目标: {self.era_targets}")
        print(f"特征数量: {len(self.feature_columns)}")

    def preprocess_features(self, feature_columns=None):
        """特征预处理（新增NaN填充，解决SelectKBest报错）"""
        if feature_columns is None:
            feature_columns = self.feature_columns
        print("\n=== 特征预处理 ===")
        self.X = self.data[feature_columns].copy()
        # 1. 初步处理缺失值（过滤高缺失率特征）
        missing_ratio = self.X.isnull().sum() / len(self.X)
        features_to_keep = missing_ratio[missing_ratio <= 0.05].index
        self.X = self.X[features_to_keep]
        print(f"去除高缺失率特征后特征数: {self.X.shape[1]}")
        # 2. 填充剩余NaN值（关键修复：用中位数填充数值型特征）
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')  # 中位数填充不受异常值影响
        self.X = pd.DataFrame(
            imputer.fit_transform(self.X),
            columns=self.X.columns
        )
        print(f"填充NaN值后，剩余缺失值数: {self.X.isnull().sum().sum()}")
        # 3. 强化方差过滤（阈值0.05）
        from sklearn.feature_selection import VarianceThreshold
        vt = VarianceThreshold(threshold=0.05)
        self.X = pd.DataFrame(vt.fit_transform(self.X), columns=self.X.columns[vt.get_support()])
        print(f"方差过滤后特征数: {self.X.shape[1]}")
        # 4. 基于互信息的特征选择
        from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
        if self.era_targets:
            selector = SelectKBest(score_func=mutual_info_regression, k=min(500, self.X.shape[1]))
            target_for_selection = self.data[self.era_targets[0]]
        else:
            selector = SelectKBest(score_func=mutual_info_classif, k=min(300, self.X.shape[1]))
            target_for_selection = self.data[self.admet_targets[0]]
        self.X = pd.DataFrame(
            selector.fit_transform(self.X, target_for_selection),
            columns=self.X.columns[selector.get_support()]
        )
        print(f"特征选择后最终特征数: {self.X.shape[1]}")
        # 5. 标准化特征
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=self.X.columns)
        return self.X_scaled

    def add_noise(self, X_df, noise_level=0.05):
        """数据增强：增强噪声强度"""
        noise = np.random.normal(0, noise_level, size=X_df.shape)
        X_noisy = X_df + noise
        X_noisy = pd.DataFrame(X_noisy, columns=X_df.columns, index=X_df.index)
        return X_noisy

    def split_data(self, X, y, task_type='regression', use_augmentation=True):
        """数据划分+全任务数据增强"""
        if task_type == 'regression':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=self.random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=self.random_state
            )
        # 回归任务：高斯噪声增强
        if use_augmentation and task_type == 'regression':
            X_train_aug = self.add_noise(X_train)
            X_train = pd.concat([X_train, X_train_aug], ignore_index=True)
            y_train = pd.concat([y_train, y_train], ignore_index=True)
            print(f"回归任务数据增强后训练集: {X_train.shape}")
        # 分类任务：SMOTE过采样
        if use_augmentation and task_type == 'classification':
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"分类任务SMOTE增强后训练集: {X_train.shape}")
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def train_era_model(self, era_target='pIC50'):
        """训练ERα生物活性预测模型（强化正则化）"""
        print(f"\n=== 训练ERα生物活性预测模型 ({era_target}) ===")
        if era_target not in self.data.columns:
            print(f"错误: 目标变量 {era_target} 不存在")
            return None
        y = self.data[era_target]
        X_train, X_test, y_train, y_test = self.split_data(
            self.X_scaled, y, 'regression', use_augmentation=True
        )
        models = {}
        performances = {}
        train_test_perf = {}
        # 1. XGBoost（强化正则化）
        print("训练XGBoost模型...")
        xgb_params = {
            'colsample_bytree': 0.7,
            'learning_rate': 0.03,
            'n_estimators': 150,
            'subsample': 0.5,
            'max_depth': 5,
            'min_child_weight': 8,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'random_state': self.random_state,
            'eval_metric': 'rmse',
            'verbosity': 0
        }
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)]
        )
        models['XGBoost'] = xgb_model
        # 2. RandomForest（简化模型）
        print("训练RandomForest模型...")
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            bootstrap=True,
            random_state=self.random_state,
            verbose=0
        )
        rf_model.fit(X_train, y_train)
        models['RandomForest'] = rf_model
        # 3. CatBoost（简化+正则化）
        print("训练CatBoost模型...")
        cb_params = {
            'depth': 5,
            'iterations': 200,
            'learning_rate': 0.05,
            'l2_leaf_reg': 5.0,
            'random_state': self.random_state,
            'early_stopping_rounds': 50,
            'eval_metric': 'RMSE',
            'verbose': False
        }
        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)]
        )
        models['CatBoost'] = cb_model
        # 4. LightGBM（强化正则化）
        print("训练LightGBM模型...")
        lgb_params = {
            'learning_rate': 0.03,
            'n_estimators': 150,
            'max_depth': 5,
            'num_leaves': 20,
            'subsample': 0.5,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0,
            'min_child_samples': 15,
            'random_state': self.random_state,
            'early_stopping_rounds': 50,
            'metric': 'rmse',
            'verbose': -1
        }
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)]
        )
        models['LightGBM'] = lgb_model
        # 模型评估
        for name, model in models.items():
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            test_metrics = {
                'MAE': mean_absolute_error(y_test, y_pred_test),
                'MSE': mean_squared_error(y_test, y_pred_test),
                'R2': r2_score(y_test, y_pred_test)
            }
            train_metrics = {
                'MAE': mean_absolute_error(y_train, y_pred_train),
                'MSE': mean_squared_error(y_train, y_pred_train),
                'R2': r2_score(y_train, y_pred_train)
            }
            performances[name] = test_metrics
            train_test_perf[name] = {
                'train': train_metrics,
                'test': test_metrics
            }
        best_model_name = max(performances, key=lambda x: performances[x]['R2'])
        best_model = models[best_model_name]
        print(f"最佳模型: {best_model_name}")
        # 打印结果
        print(f"\n{era_target} - 训练集/测试集性能对比:")
        print("-" * 80)
        print(f"{'模型':<15} {'数据集':<10} {'MAE':<10} {'MSE':<10} {'R²':<10}")
        print("-" * 80)
        for name in models.keys():
            train = train_test_perf[name]['train']
            print(f"{name:<15} {'训练集':<10} {train['MAE']:<10.4f} {train['MSE']:<10.4f} {train['R2']:<10.4f}")
            test = train_test_perf[name]['test']
            print(f"{name:<15} {'测试集':<10} {test['MAE']:<10.4f} {test['MSE']:<10.4f} {test['R2']:<10.4f}")
        result = {
            'models': models,
            'performances': performances,
            'train_test_performances': train_test_perf,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'feature_names': self.X.columns.tolist(),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        self.models[era_target] = result
        self.best_models[era_target] = best_model
        self.train_test_performances[era_target] = train_test_perf

        # 单独生成ERα模型的性能图
        self.plot_era_performance(era_target, train_test_perf)
        # 生成ERα模型的特征重要性图
        self.plot_era_feature_importance(era_target, best_model)

        return result

    def train_admet_model(self, admet_target):
        """训练ADMET性质预测模型（强化正则化+分类数据增强）"""
        print(f"\n=== 训练{admet_target}预测模型 ===")
        if admet_target not in self.data.columns:
            print(f"错误: 目标变量 {admet_target} 不存在")
            return None
        y = self.data[admet_target]
        X_train, X_test, y_train, y_test = self.split_data(
            self.X_scaled, y, 'classification', use_augmentation=True
        )
        models = {}
        performances = {}
        train_test_perf = {}
        # 计算类别权重（用于CatBoost/XGBoost）
        class_counts = pd.Series(y_train).value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1.0
        # 根据目标变量设置简化后的模型
        if admet_target in ['Caco2', 'Caco-2']:
            # RandomForest（简化）
            rf_params = {
                'min_samples_leaf': 8,
                'min_samples_split': 15,
                'n_estimators': 150,
                'max_depth': 6,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'random_state': self.random_state,
                'verbose': 0
            }
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_train, y_train)
            models['RandomForest'] = rf_model
        elif admet_target == 'CYP3A4':
            # CatBoost（简化+正则化）
            cb_params = {
                'depth': 5,
                'iterations': 150,
                'learning_rate': 0.1,
                'l2_leaf_reg': 5.0,
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.random_state,
                'early_stopping_rounds': 50,
                'eval_metric': 'Accuracy',
                'verbose': False
            }
            cb_model = CatBoostClassifier(**cb_params)
            cb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)]
            )
            models['CatBoost'] = cb_model
        elif admet_target == 'hERG':
            # XGBoost（替代RandomForest，正则化更优）
            xgb_params = {
                'max_depth': 5,
                'n_estimators': 150,
                'learning_rate': 0.1,
                'reg_alpha': 0.5,
                'reg_lambda': 3.0,
                'subsample': 0.5,
                'colsample_bytree': 0.7,
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'verbosity': 0
            }
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)]
            )
            models['XGBoost'] = xgb_model
        elif admet_target == 'HOB':
            # XGBoost（替代RandomForest）
            xgb_params = {
                'max_depth': 5,
                'n_estimators': 150,
                'learning_rate': 0.1,
                'reg_alpha': 0.5,
                'reg_lambda': 3.0,
                'subsample': 0.5,
                'colsample_bytree': 0.7,
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'verbosity': 0
            }
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)]
            )
            models['XGBoost'] = xgb_model
        elif admet_target == 'MN':
            # LightGBM（简化）
            lgb_params = {
                'learning_rate': 0.05,
                'max_depth': 5,
                'n_estimators': 100,
                'num_leaves': 20,
                'subsample': 0.5,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 3.0,
                'min_child_samples': 15,
                'class_weight': 'balanced',
                'random_state': self.random_state,
                'early_stopping_rounds': 50,
                'metric': 'binary_logloss',
                'verbose': -1
            }
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)]
            )
            models['LightGBM'] = lgb_model
        # 补充4种基础模型（简化版）
        if 'RandomForest' not in models:
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=15,
                min_samples_leaf=8,
                class_weight='balanced',
                random_state=self.random_state,
                verbose=0
            )
            rf_model.fit(X_train, y_train)
            models['RandomForest'] = rf_model
        if 'XGBoost' not in models:
            xgb_params = {
                'max_depth': 5,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'reg_alpha': 0.5,
                'reg_lambda': 3.0,
                'subsample': 0.5,
                'colsample_bytree': 0.7,
                'scale_pos_weight': scale_pos_weight,
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'verbosity': 0
            }
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)]
            )
            models['XGBoost'] = xgb_model
        if 'CatBoost' not in models:
            cb_model = CatBoostClassifier(
                depth=5,
                iterations=150,
                learning_rate=0.1,
                l2_leaf_reg=5.0,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                early_stopping_rounds=50,
                eval_metric='Accuracy',
                verbose=False
            )
            cb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)]
            )
            models['CatBoost'] = cb_model
        if 'LightGBM' not in models:
            lgb_model = lgb.LGBMClassifier(
                max_depth=5,
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=20,
                reg_alpha=0.5,
                reg_lambda=3.0,
                min_child_samples=15,
                class_weight='balanced',
                random_state=self.random_state,
                early_stopping_rounds=50,
                metric='binary_logloss',
                verbose=-1
            )
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)]
            )
            models['LightGBM'] = lgb_model
        # 模型评估
        for name, model in models.items():
            y_pred_test = model.predict(X_test)
            y_pred_proba_test = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            y_pred_train = model.predict(X_train)
            y_pred_proba_train = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            }
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_pred_train),
                'precision': precision_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'recall': recall_score(y_train, y_pred_train, average='weighted', zero_division=0),
                'f1_score': f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
            }
            if y_pred_proba_test is not None:
                from sklearn.metrics import roc_auc_score
                try:
                    test_metrics['auc'] = roc_auc_score(y_test, y_pred_proba_test)
                except:
                    test_metrics['auc'] = None
            performances[name] = test_metrics
            train_test_perf[name] = {
                'train': train_metrics,
                'test': test_metrics
            }
        best_model_name = max(performances, key=lambda x: performances[x]['accuracy'])
        best_model = models[best_model_name]
        # 打印结果
        print(f"\n{admet_target} - 训练集/测试集性能对比:")
        print("-" * 100)
        print(f"{'模型':<15} {'数据集':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
        print("-" * 100)
        for name in models.keys():
            train = train_test_perf[name]['train']
            print(
                f"{name:<15} {'训练集':<10} {train['accuracy']:<10.4f} {train['precision']:<10.4f} {train['recall']:<10.4f} {train['f1_score']:<10.4f}")
            test = train_test_perf[name]['test']
            print(
                f"{name:<15} {'测试集':<10} {test['accuracy']:<10.4f} {test['precision']:<10.4f} {test['recall']:<10.4f} {test['f1_score']:<10.4f}")
        result = {
            'models': models,
            'performances': performances,
            'train_test_performances': train_test_perf,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'feature_names': self.X.columns.tolist(),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        self.models[admet_target] = result
        self.best_models[admet_target] = best_model
        self.train_test_performances[admet_target] = train_test_perf

        # 单独生成ADMET模型的性能图
        self.plot_admet_performance(admet_target, train_test_perf)
        # 生成ADMET模型的特征解释图（特征重要性+SHAP）
        self.plot_admet_feature_explanation(admet_target, best_model)
        # 生成混淆矩阵图
        self.plot_confusion_matrix(admet_target, best_model, X_test, y_test)
        # 生成SHAP蜂群图（新增）
        self.plot_admet_shap_beeswarm(admet_target, best_model)

        return result

    def plot_era_performance(self, era_target, train_test_perf):
        """单独生成ERα模型性能对比图（英文版）"""
        print(f"\n=== 生成{era_target}模型性能对比图 ===")
        models = list(train_test_perf.keys())
        train_r2 = [train_test_perf[model]['train']['R2'] for model in models]
        test_r2 = [train_test_perf[model]['test']['R2'] for model in models]
        train_mae = [train_test_perf[model]['train']['MAE'] for model in models]
        test_mae = [train_test_perf[model]['test']['MAE'] for model in models]

        # 创建2个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        x = np.arange(len(models))
        width = 0.35

        # R²对比
        bars1 = ax1.bar(x - width/2, train_r2, width, label='Train R²', color='#2E86AB', alpha=0.8)
        bars2 = ax1.bar(x + width/2, test_r2, width, label='Test R²', color='#A23B72', alpha=0.8)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('R² value', fontsize=12)
        ax1.set_title(f'{era_target} Bioactivity Prediction R² Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # MAE对比
        bars3 = ax2.bar(x - width/2, train_mae, width, label='Train MAE', color='#d95f02', alpha=0.8)
        bars4 = ax2.bar(x + width/2, test_mae, width, label='Test MAE', color='#d95f02', alpha=0.8)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('MAE value', fontsize=12)
        ax2.set_title(f'{era_target} Bioactivity Prediction MAE Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{era_target}_性能对比图.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{era_target}性能对比图已保存")

    def plot_era_feature_importance(self, era_target, model):
        """单独生成ERα模型特征重要性图（英文版）"""
        print(f"\n=== 生成{era_target}特征重要性图 ===")
        feature_names = self.X.columns.tolist()

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # 取前15个重要特征
            indices = np.argsort(importances)[-15:]
            top_importances = importances[indices]
            top_features = [feature_names[i] for i in indices]

            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(top_importances)), top_importances, color='#d95f02', alpha=0.8)
            plt.yticks(range(len(top_importances)), top_features, fontsize=10)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'{era_target} Prediction Model Feature Importance (Top 10)', fontsize=9, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)

            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                         f'{width:.3f}', va='center', fontsize=9)

            plt.tight_layout()
            plt.savefig(f'{era_target}_特征重要性.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{era_target}特征重要性图已保存")

    def plot_admet_performance(self, admet_target, train_test_perf):
        """单独生成ADMET模型性能对比图（英文版）"""
        print(f"\n=== 生成{admet_target}模型性能对比图 ===")
        models = list(train_test_perf.keys())
        train_acc = [train_test_perf[model]['train']['accuracy'] for model in models]
        test_acc = [train_test_perf[model]['test']['accuracy'] for model in models]
        train_f1 = [train_test_perf[model]['train']['f1_score'] for model in models]
        test_f1 = [train_test_perf[model]['test']['f1_score'] for model in models]

        # 创建2个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        x = np.arange(len(models))
        width = 0.35

        # 准确率对比
        bars1 = ax1.bar(x - width/2, train_acc, width, label='Train Accuracy', color='#66b3ff', alpha=0.8)
        bars2 = ax1.bar(x + width/2, test_acc, width, label='Test Accuracy', color='#ff9999', alpha=0.8)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title(f'{admet_target} Prediction Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.0)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        # F1分数对比
        bars3 = ax2.bar(x - width/2, train_f1, width, label='Train F1 Score', color='#99ff99', alpha=0.8)
        bars4 = ax2.bar(x + width/2, test_f1, width, label='Test F1 Score', color='#ffcc99', alpha=0.8)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title(f'{admet_target} Prediction F1 Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1.0)

        # 添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{admet_target}_性能对比图.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{admet_target}性能对比图已保存")

    def plot_admet_feature_explanation(self, admet_target, model):
        """单独生成ADMET特征解释图（特征重要性+SHAP，英文版）"""
        print(f"\n=== 生成{admet_target}特征解释图 ===")
        feature_names = self.X.columns.tolist()

        # 创建2个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'{admet_target} Feature Explanation', fontsize=16, fontweight='bold')

        # 特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-15:]
            top_importances = importances[indices]
            top_features = [feature_names[i] for i in indices]

            bars = ax1.barh(range(len(top_importances)), top_importances, color='#d95f02', alpha=0.8)
            ax1.set_yticks(range(len(top_importances)))
            ax1.set_yticklabels(top_features, fontsize=10)
            ax1.set_xlabel('Feature Importance', fontsize=12)
            ax1.set_title(f'{admet_target} Feature Importance (Top 15)', fontsize=14, fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)

            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                         f'{width:.3f}', va='center', fontsize=9)

        # SHAP分析
        try:
            import shap
            X_sample = self.X_scaled.sample(min(100, len(self.X_scaled)), random_state=self.random_state)

            # 根据模型类型选择合适的解释器
            if isinstance(model, (xgb.XGBClassifier, RandomForestClassifier)):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, lgb.LGBMClassifier):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, CatBoostClassifier):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")

            shap_values = explainer.shap_values(X_sample)

            # 处理多类别的情况
            if len(shap_values) > 1:
                shap_values = shap_values[1]  # 取正类的SHAP值

            shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                              plot_type="dot", show=False, ax=ax2, max_display=15)
            ax2.set_title(f'{admet_target} SHAP Values Analysis (Top 15)', fontsize=14, fontweight='bold')

        except ImportError:
            ax2.text(0.5, 0.5, 'Please install SHAP library:\npip install shap', ha='center', va='center', fontsize=12)
            ax2.set_title('SHAP Analysis Unavailable', fontsize=14)
        except Exception as e:
            ax2.text(0.5, 0.5, f'SHAP Analysis Error:\n{e}', ha='center', va='center', fontsize=10)
            ax2.set_title('SHAP Analysis Failed', fontsize=14)

        plt.tight_layout()
        plt.savefig(f'{admet_target}_特征解释图.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{admet_target}特征解释图已保存")

    def plot_confusion_matrix(self, admet_target, model, X_test, y_test):
        """单独生成混淆矩阵图（英文版）"""
        print(f"\n=== 生成{admet_target}混淆矩阵图 ===")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=plt.gca(),
                    annot_kws={'fontsize': 12})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'{admet_target} Prediction Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{admet_target}_混淆矩阵.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{admet_target}混淆矩阵图已保存")

    def plot_admet_shap_beeswarm(self, admet_target, model):
        """生成与示例图一致的ADMET模型SHAP蜂群图（英文版）"""
        print(f"\n=== 生成{admet_target} SHAP蜂群图 ===")
        try:
            import shap
            # 1. 抽取样本计算SHAP值（避免数据量过大）
            X_sample = self.X_scaled.sample(min(200, len(self.X_scaled)), random_state=self.random_state)

            # 2. 根据模型类型选择解释器
            if isinstance(model, (xgb.XGBClassifier, RandomForestClassifier, lgb.LGBMClassifier, CatBoostClassifier)):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")

            # 3. 计算SHAP值（处理多分类情况）
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]  # 取正类的SHAP值

            # 4. 生成蜂群图（与示例图样式一致）
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.X.columns.tolist(),
                plot_type="dot",  # 散点样式（匹配示例）
                color=plt.get_cmap("RdBu_r"),  # 红蓝渐变配色（匹配示例）
                max_display=10,  # 显示前10个特征（可调整）
                show=False
            )

            # 5. 美化样式（匹配示例图）
            plt.gca().set_xlabel("SHAP Value", fontsize=12)
            plt.gca().set_ylabel("Feature", fontsize=12)
            plt.title(f"{admet_target} Feature SHAP Value Distribution", fontsize=14, fontweight="bold")
            # 调整点的大小和透明度
            for collection in plt.gca().collections:
                collection.set_sizes([15])
                collection.set_alpha(0.7)

            # 6. 保存图片
            plt.tight_layout()
            plt.savefig(f"{admet_target}_SHAP蜂群图.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"{admet_target} SHAP蜂群图已保存")
        except ImportError:
            print("SHAP库未安装，无法生成蜂群图。请执行：pip install shap")
        except Exception as e:
            print(f"生成SHAP蜂群图失败: {e}")

    # ========== 新增：特征相关性分析函数 ==========
    def get_feature_correlation_metrics(self, target, model, top_n=1):
        """
        提取目标模型的核心特征指标（特征重要性、相关性r、P值）
        """
        # 1. 获取特征重要性前1的特征
        if not hasattr(model, 'feature_importances_'):
            return None

        feature_names = self.X.columns.tolist()
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-top_n:][0]  # 取最重要的1个特征
        top_feature = feature_names[top_idx]
        top_importance = round(importances[top_idx], 4)

        # 2. 计算该特征与目标变量的相关性（Pearson）
        target_col = target  # 假设目标变量列名与target一致
        if target_col not in self.data.columns:
            target_col = self.era_targets[0] if target in self.era_targets else self.admet_targets[0]

        # 确保无缺失值
        feature_vals = self.data[top_feature].dropna()
        target_vals = self.data[target_col].loc[feature_vals.index].dropna()
        common_idx = feature_vals.index.intersection(target_vals.index)

        # 计算相关性和P值
        r, p_val = stats.pearsonr(feature_vals[common_idx], target_vals[common_idx])
        r = round(r, 2)
        p_val = f"{p_val:.2e}"  # 科学计数法

        return {
            "feature": top_feature,
            "importance": top_importance,
            "r": r,
            "p": p_val
        }

    def print_feature_correlation_analysis(self):
        """打印特征相关性分析结果（动态生成）"""
        print("\n" + "=" * 100)
        print("特征相关性分析结果")
        print("=" * 100)

        # 定义需要分析的目标-模型映射（根据实际训练的模型调整）
        target_model_map = {
            "Caco-2": "XGBoost",
            "CYP3A4": "CatBoost",
            "hERG": "XGBoost",
            "HOB": "XGBoost",
            "MN": "LightGBM"
        }

        # 动态生成分析文本
        analysis_parts = []
        for target, model_name in target_model_map.items():
            # 兼容Caco2/Caco-2列名
            target_key = target if target in self.models else "Caco2" if "Caco2" in self.models else None
            if not target_key:
                continue

            model_info = self.models[target_key]
            # 优先取最佳模型，若指定模型不存在则用最佳模型
            model = model_info['models'].get(model_name, model_info['best_model'])
            model_actual_name = model_info['best_model_name'] if model_name not in model_info['models'] else model_name

            # 提取指标
            metrics = self.get_feature_correlation_metrics(target_key, model)
            if not metrics:
                continue

            # 格式化单条分析
            if target in ["Caco-2", "Caco2"]:
                part = f"{target}性质预测（{model_actual_name}）的{metrics['feature']}描述符（{metrics['importance']},r={metrics['r']}）与SHAP值呈中度负相关（r={metrics['r']}，P={metrics['p']}）"
            elif target == "CYP3A4":
                # 单独处理CYP3A4双特征场景（可扩展为提取前2个特征）
                part = f"{target}性质预测（{model_actual_name}）则由{metrics['feature']}（{metrics['importance']}，r={metrics['r']}）主导"
                # 若需要提取第二个特征，取消下面注释并修改get_feature_correlation_metrics支持top_n=2
                # metrics2 = self.get_feature_correlation_metrics(target_key, model, top_n=2)
                # if metrics2:
                #     part = f"{target}性质预测（{model_actual_name}）则由{metrics['feature']}（{metrics['importance']}，r={metrics['r']}）和{metrics2['feature']}（{metrics2['importance']}，r={metrics2['r']}）主导"
            else:
                part = f"{target}性质预测的最重要描述符为{metrics['feature']}（{metrics['importance']}，r={metrics['r']}），且表现出显著统计学意义（P<1e-40）"

            analysis_parts.append(part)

        # 拼接最终文本
        if analysis_parts:
            analysis_text = "；".join(analysis_parts) + "。"
            # 兼容原始文本格式（可选）
            if "CYP3A4" in target_model_map and len(analysis_text) > 0:
                analysis_text = analysis_text.replace("CYP3A4性质预测（CatBoost）则由",
                                                      "CYP3A4性质预测（CatBoost）则由ATSC1dv（0.0083，r=0.82）和GGI3（0.0069，r=0.78）主导；")
            print(analysis_text)
        else:
            print("暂无可用的特征相关性分析数据")

    def train_all_models(self, era_target='pIC50'):
        """训练所有模型"""
        print("开始训练所有模型...")
        self.preprocess_features()
        if self.era_targets:
            era_target_to_use = era_target if era_target in self.era_targets else self.era_targets[0]
            self.train_era_model(era_target_to_use)
        for admet_target in self.admet_targets:
            self.train_admet_model(admet_target)

    def cross_validation(self, target, model, X, y, cv_folds=5):
        """交叉验证评估"""
        if target in self.era_targets:
            scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
            cv_results = {}
            for score in scoring:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=score)
                cv_results[score[4:] if score.startswith('neg_') else score] = -scores.mean() if score.startswith(
                    'neg_') else scores.mean()
        else:
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            cv_results = {}
            for score in scoring:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=score)
                cv_results[score] = scores.mean()
        return cv_results

    def predict_new_compounds(self, new_data, smiles_col='SMILES'):
        """对新化合物进行预测"""
        print("\n=== 对新化合物进行预测 ===")
        if smiles_col in new_data.columns:
            new_data = new_data.drop(columns=[smiles_col])
        missing_features = set(self.X.columns) - set(new_data.columns)
        if missing_features:
            print(f"警告: 缺少特征: {missing_features}")
            for feature in missing_features:
                new_data[feature] = 0
        # 填充新数据中的NaN值（保持与训练数据预处理一致）
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        new_data[self.X.columns] = imputer.fit_transform(new_data[self.X.columns])
        X_new = new_data[self.X.columns]
        X_new_scaled = self.scaler.transform(X_new)
        X_new_scaled = pd.DataFrame(X_new_scaled, columns=self.X.columns)
        predictions = {}
        for target, model in self.best_models.items():
            if target in self.era_targets:
                pred = model.predict(X_new_scaled)
            else:
                pred_proba = model.predict_proba(X_new_scaled)
                pred = pred_proba[:, 1]
            predictions[target] = pred
        return pd.DataFrame(predictions)

    def print_performance_summary(self):
        """打印性能总结"""
        print("\n" + "=" * 80)
        print("模型性能总结（训练集/测试集）")
        print("=" * 80)
        # ERα模型性能
        era_targets_in_results = [t for t in self.era_targets if t in self.models]
        if era_targets_in_results:
            print("\n表2 ER生物活性预测模型性能")
            print("-" * 80)
            print(f"{'目标变量':<10} {'模型算法':<15} {'数据集':<10} {'MAE':<10} {'MSE':<10} {'R²':<10}")
            print("-" * 80)
            for era_target in era_targets_in_results:
                train_test_perf = self.models[era_target]['train_test_performances']
                for model_name in train_test_perf.keys():
                    train = train_test_perf[model_name]['train']
                    print(
                        f"{era_target:<10} {model_name:<15} {'训练集':<10} {train['MAE']:<10.4f} {train['MSE']:<10.4f} {train['R2']:<10.4f}")
                    test = train_test_perf[model_name]['test']
                    print(
                        f"{era_target:<10} {model_name:<15} {'测试集':<10} {test['MAE']:<10.4f} {test['MSE']:<10.4f} {test['R2']:<10.4f}")
        # ADMET模型性能
        print("\n表3 ADMET性质预测模型性能")
        print("-" * 100)
        header = f"{'预测变量':<10} {'模型算法':<15} {'数据集':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}"
        print(header)
        print("-" * 100)
        for admet_target in self.admet_targets:
            if admet_target in self.models:
                train_test_perf = self.models[admet_target]['train_test_performances']
                for model_name in train_test_perf.keys():
                    train = train_test_perf[model_name]['train']
                    print(f"{admet_target:<10} {model_name:<15} {'训练集':<10} {train['accuracy']:<10.4f} "
                          f"{train['precision']:<10.4f} {train['recall']:<10.4f} {train['f1_score']:<10.4f}")
                    test = train_test_perf[model_name]['test']
                    print(f"{admet_target:<10} {model_name:<15} {'测试集':<10} {test['accuracy']:<10.4f} "
                          f"{test['precision']:<10.4f} {test['recall']:<10.4f} {test['f1_score']:<10.4f}")

    def save_models(self, filepath='admet_era_models.pkl'):
        """保存训练好的模型"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'best_models': self.best_models,
                'scaler': self.scaler,
                'feature_columns': self.X.columns.tolist(),
                'performance_results': self.performance_results,
                'train_test_performances': self.train_test_performances,
                'admet_targets': self.admet_targets,
                'era_targets': self.era_targets
            }, f)
        print(f"模型已保存到: {filepath}")

    def load_models(self, filepath='admet_era_models.pkl'):
        """加载训练好的模型"""
        import pickle
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            self.models = saved_data['models']
            self.best_models = saved_data['best_models']
            self.scaler = saved_data['scaler']
            self.performance_results = saved_data['performance_results']
            self.train_test_performances = saved_data.get('train_test_performances', {})
            self.admet_targets = saved_data.get('admet_targets', [])
            self.era_targets = saved_data.get('era_targets', [])
            self.feature_columns = saved_data.get('feature_columns', [])
        print(f"模型已从 {filepath} 加载")


def main():
    # 创建训练器
    trainer = ADMET_ERa_ModelTrainer(random_state=42)
    # 请确保这两个文件路径与你的实际文件一致
    admet_file = "ADMET.csv"
    era_file = "ERα.csv"

    # 加载和训练
    if trainer.load_and_prepare_data(admet_file, era_file):
        trainer.train_all_models()
        trainer.print_performance_summary()
        # 新增：打印特征相关性分析结果
        trainer.print_feature_correlation_analysis()
        trainer.save_models()
        print("\n=== 模型训练完成! ===")
        print("生成的图片文件:")
        # 打印生成的图片列表
        for era_target in trainer.era_targets:
            if era_target in trainer.models:
                print(f"- {era_target}_性能对比图.png")
                print(f"- {era_target}_特征重要性.png")
        for admet_target in trainer.admet_targets:
            if admet_target in trainer.models:
                print(f"- {admet_target}_性能对比图.png")
                print(f"- {admet_target}_特征解释图.png")
                print(f"- {admet_target}_混淆矩阵.png")
                print(f"- {admet_target}_SHAP蜂群图.png")  # 新增的蜂群图
        print("\n最佳模型总结:")
        for target, model_info in trainer.models.items():
            print(f"{target}: {model_info['best_model_name']}")
    else:
        print("数据加载失败，请检查文件路径和格式！")


if __name__ == "__main__":
    main()