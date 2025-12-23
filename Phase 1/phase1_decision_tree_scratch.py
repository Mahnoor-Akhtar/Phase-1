"""
COMPLETE MACHINE LEARNING PROJECT WITH VISUALIZATIONS
Diabetes Prediction - All Models Comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              GradientBoostingClassifier)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - skipping XGBoost model")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available - skipping LightGBM model")

# Try to import CatBoost, skip if not available
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available - skipping CatBoost model")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("=" * 100)
print("COMPLETE ML PROJECT: DIABETES PREDICTION WITH VISUALIZATIONS")
print("=" * 100)

# ==================== 1. DATASET CREATION & VISUALIZATION ====================

def create_and_visualize_dataset():
    """Create dataset and visualize it"""
    print("\n📊 CREATING AND VISUALIZING DATASET")
    print("-" * 50)
    
    np.random.seed(42)
    n_samples = 10000
    
    # Create dataset
    data = {
        'gender': np.random.choice(['Female', 'Male'], n_samples, p=[0.5, 0.5]),
        'age': np.random.normal(50, 15, n_samples).clip(20, 80),
        'hypertension': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'heart_disease': np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
        'smoking_history': np.random.choice(['never', 'former', 'current'], n_samples),
        'bmi': np.random.normal(28, 6, n_samples).clip(15, 50),
        'HbA1c_level': np.random.normal(5.5, 1.5, n_samples).clip(3.5, 12),
        'blood_glucose_level': np.random.normal(120, 40, n_samples).clip(70, 300),
    }
    
    df = pd.DataFrame(data)
    
    # Create target
    risk_score = (
        0.3 * (df['age'] - 20) / 60 +
        0.4 * (df['blood_glucose_level'] - 70) / 230 +
        0.4 * (df['HbA1c_level'] - 3.5) / 8.5 +
        0.2 * (df['bmi'] - 15) / 35 +
        0.1 * df['hypertension'] +
        0.1 * df['heart_disease']
    )
    
    diabetes_proba = 1 / (1 + np.exp(-(risk_score - 0.5) * 5))
    df['diabetes'] = np.random.binomial(1, diabetes_proba)
    
    print(f"✅ Dataset created: {n_samples:,} samples")
    
    # Visualization 1: Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    class_counts = df['diabetes'].value_counts()
    colors = ['#66c2a5', '#fc8d62']
    axes[0].bar(['Non-Diabetic', 'Diabetic'], class_counts.values, color=colors)
    axes[0].set_title('Diabetes Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, count in enumerate(class_counts.values):
        axes[0].text(i, count + 50, f'{count}\n({count/n_samples*100:.1f}%)', 
                    ha='center', fontsize=10)
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=['Non-Diabetic', 'Diabetic'], 
                autopct='%1.1f%%', colors=colors, startangle=90,
                textprops={'fontsize': 11})
    axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
    
    plt.suptitle('Dataset Class Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.close() # Don't block
    # plt.show()
    
    # Visualization 2: Feature Distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    for idx, feature in enumerate(numeric_features[:4]):
        sns.histplot(data=df, x=feature, hue='diabetes', ax=axes[idx], 
                    kde=True, palette=colors, alpha=0.6)
        axes[idx].set_title(f'{feature} Distribution', fontsize=12)
        axes[idx].set_xlabel('')
    
    # Categorical features
    categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
    
    for idx, feature in enumerate(categorical_features):
        ax_idx = idx + 4
        cross_tab = pd.crosstab(df[feature], df['diabetes'], normalize='index')
        cross_tab.plot(kind='bar', stacked=True, ax=axes[ax_idx], color=colors)
        axes[ax_idx].set_title(f'{feature} vs Diabetes', fontsize=12)
        axes[ax_idx].set_xlabel('')
        axes[ax_idx].legend(['Non-Diabetic', 'Diabetic'])
    
    plt.suptitle('Feature Distributions by Diabetes Status', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.close() # Don't block
    # plt.show()
    
    # Visualization 3: Correlation Matrix
    df_encoded = df.copy()
    df_encoded['gender'] = df_encoded['gender'].map({'Female': 0, 'Male': 1})
    df_encoded['smoking_history'] = LabelEncoder().fit_transform(df_encoded['smoking_history'])
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_encoded.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.close() # Don't block
    # plt.show()
    
    return df

# ==================== 2. DATA PREPROCESSING ====================

def preprocess_data(df):
    """Preprocess the dataset"""
    print("\n🔧 PREPROCESSING DATA")
    print("-" * 50)
    
    df_processed = df.copy()
    
    # Encode categorical variables
    df_processed['gender'] = df_processed['gender'].map({'Female': 0, 'Male': 1})
    
    smoking_mapping = {'never': 0, 'former': 1, 'current': 2}
    df_processed['smoking_history'] = df_processed['smoking_history'].map(smoking_mapping)
    
    # Separate features and target
    X = df_processed.drop('diabetes', axis=1)
    y = df_processed['diabetes']
    
    feature_names = X.columns.tolist()
    
    print(f"✅ Preprocessing complete")
    print(f"   Features: {len(feature_names)}")
    print(f"   Samples: {len(X):,}")
    
    return X.values, y.values, feature_names, df_processed

# ==================== 3. DECISION TREE FROM SCRATCH ====================

class TreeNode:
    """Decision Tree Node"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeScratch:
    """Decision Tree from Scratch"""
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def entropy(self, y):
        """Calculate entropy"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / len(y)
        entropy_val = 0
        for p in probs:
            if p > 0:
                entropy_val -= p * np.log2(p)
        return entropy_val
    
    def info_gain(self, X_col, y, threshold):
        """Calculate information gain"""
        left_mask = X_col <= threshold
        right_mask = X_col > threshold
        
        if sum(left_mask) == 0 or sum(right_mask) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = sum(left_mask), sum(right_mask)
        
        gain = self.entropy(y) - (
            (n_left/n) * self.entropy(y[left_mask]) + 
            (n_right/n) * self.entropy(y[right_mask])
        )
        return gain
    
    def best_split(self, X, y):
        """Find best split"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self.info_gain(X[:, feature_idx], y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """Build tree recursively"""
        n_samples = X.shape[0]
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return TreeNode(value=np.bincount(y).argmax())
        
        feature, threshold, gain = self.best_split(X, y)
        
        if gain <= 0:
            return TreeNode(value=np.bincount(y).argmax())
        
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        
        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(feature=feature, threshold=threshold, left=left, right=right)
    
    def fit(self, X, y):
        """Train the tree"""
        self.root = self.build_tree(X, y)
    
    def predict_one(self, x, node):
        """Predict single sample"""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_one(x, self.root) for x in X])

# ==================== 4. ALL MODELS TRAINING ====================

def train_and_evaluate_models(X_train, y_train, X_test, y_test, feature_names):
    """Train all models and return results"""
    print("\n🚀 TRAINING ALL MODELS")
    print("-" * 50)
    
    models = {}
    results = []
    predictions = {}
    prob_predictions = {}
    
    # Model configurations
    model_configs = [
        ('Decision Tree (Scratch)', DecisionTreeScratch(max_depth=4, min_samples_split=20), 'Custom'),
        ('Decision Tree', DecisionTreeClassifier(max_depth=4, random_state=42), 'sklearn'),
        ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42), 'sklearn'),
        ('Extra Trees', ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42), 'sklearn'),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42), 'sklearn'),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42), 'sklearn'),
    ]
    
    if XGBOOST_AVAILABLE:
        model_configs.append(('XGBoost', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='logloss'), 'xgboost'))
        
    if LIGHTGBM_AVAILABLE:
        model_configs.append(('LightGBM', LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, verbose=-1), 'lightgbm'))
    
    # Add CatBoost only if available
    if CATBOOST_AVAILABLE:
        model_configs.append(('CatBoost', CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, random_state=42, verbose=0), 'catboost'))
    
    total_models = len(model_configs)
    
    for i, (name, model, library) in enumerate(model_configs):
        print(f"[{i+1}/{total_models}] Training {name}...")
        
        # Train model
        if name == 'Decision Tree (Scratch)':
            # Use subset for scratch model execution speed
            print("    (Training on subset of 1000 samples for speed)")
            indices = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
            model.fit(X_train[indices], y_train[indices])
        else:
            model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Store model and predictions
        models[name] = model
        predictions[name] = y_pred
        
        # Calculate metrics
        metrics = {
            'Model': name,
            'Library': library,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Try to get probability predictions
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            prob_predictions[name] = y_prob
        
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    print("\n✅ All models trained successfully!")
    
    return results_df, models, predictions, prob_predictions

# ==================== 5. VISUALIZATION FUNCTIONS ====================

def plot_model_comparison(results_df):
    """Plot comparison of all models"""
    print("\n📊 VISUALIZING MODEL COMPARISON")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Sort by metric
        sorted_df = results_df.sort_values(metric, ascending=True)
        
        bars = ax.barh(range(len(sorted_df)), sorted_df[metric], color=colors, edgecolor='black')
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['Model'], fontsize=9)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', va='center', fontsize=8)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved results_comparison.png")
    # plt.show()
    
    # Radar chart for best models
    plot_radar_chart(results_df)

def plot_radar_chart(results_df):
    """Plot radar chart for top 4 models"""
    print("📈 Creating Radar Chart...")
    
    # Select top 4 models by accuracy
    top_models = results_df.nlargest(4, 'Accuracy')
    
    # Metrics for radar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Normalize metrics to 0-1 scale
    normalized_data = []
    for metric in metrics:
        max_val = top_models[metric].max()
        normalized_data.append(top_models[metric].values / max_val)
    
    normalized_data = np.array(normalized_data)
    
    # Plot radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    for i, (idx, row) in enumerate(top_models.iterrows()):
        values = normalized_data[:, i].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], markersize=6)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('Top 4 Models - Radar Chart Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results_radar.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved results_radar.png")
    # plt.show()

def plot_confusion_matrices(predictions, y_test, model_names):
    """Plot confusion matrices for all models"""
    print("\n🔢 VISUALIZING CONFUSION MATRICES")
    
    n_models = len(model_names)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, model_name in enumerate(model_names):
        if idx < len(axes):
            ax = axes[idx]
            y_pred = predictions[model_name]
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'shrink': 0.8}, annot_kws={'size': 10})
            
            # Set labels
            ax.set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)
            ax.set_xticklabels(['No', 'Yes'], fontsize=8)
            ax.set_yticklabels(['No', 'Yes'], fontsize=8, rotation=0)
    
    # Hide empty subplots
    for idx in range(len(model_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved results_confusion_matrices.png")
    # plt.show()

def plot_roc_curves(prob_predictions, y_test, model_names):
    """Plot ROC curves for models with probability predictions"""
    print("\n📈 PLOTTING ROC CURVES")
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for i, model_name in enumerate(model_names):
        if model_name in prob_predictions:
            y_prob = prob_predictions[model_name]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_roc_curves.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved results_roc_curves.png")
    # plt.show()

def plot_precision_recall_curves(prob_predictions, y_test, model_names):
    """Plot Precision-Recall curves"""
    print("📊 PLOTTING PRECISION-RECALL CURVES")
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    for i, model_name in enumerate(model_names):
        if model_name in prob_predictions:
            y_prob = prob_predictions[model_name]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            avg_precision = np.mean(precision)
            
            plt.plot(recall, precision, color=colors[i], lw=2,
                    label=f'{model_name} (Avg Precision = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_pr_curves.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved results_pr_curves.png")
    # plt.show()

def plot_feature_importance(models, feature_names):
    """Plot feature importance for tree-based models"""
    print("\n🎯 VISUALIZING FEATURE IMPORTANCE")
    
    # Get models with feature_importances_
    importance_models = {name: model for name, model in models.items() 
                        if hasattr(model, 'feature_importances_')}
    
    if not importance_models:
        print("No feature importance data available")
        return
    
    n_models = len(importance_models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, (name, model) in enumerate(list(importance_models.items())[:len(axes)]):
        ax = axes[idx]
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[-10:]  # Top 10 features
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_features))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        
        ax.barh(y_pos, top_importances, color=colors, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title(f'{name}\nTop 10 Features', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(top_importances):
            ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(importance_models), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature Importance Across Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results_feature_importance.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved results_feature_importance.png")
    # plt.show()

def plot_decision_tree(models, feature_names):
    """Visualize decision tree from sklearn"""
    print("\n🌳 VISUALIZING DECISION TREE")
    
    # Find sklearn Decision Tree model
    for name, model in models.items():
        if 'Decision Tree' in name and name != 'Decision Tree (Scratch)':
            plt.figure(figsize=(20, 10))
            plot_tree(model, 
                     feature_names=feature_names,
                     class_names=['Non-Diabetic', 'Diabetic'],
                     filled=True,
                     rounded=True,
                     proportion=True,
                     fontsize=10,
                     max_depth=3)  # Show only first 3 levels for clarity
            plt.title(f'{name} - Decision Tree Visualization', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('results_decision_tree_viz.png', dpi=300, bbox_inches='tight')
            print("   ✅ Saved results_decision_tree_viz.png")
            # plt.show()
            break

def plot_model_training_time(results_df, execution_times):
    """Plot model training times"""
    print("\n⏱️ VISUALIZING TRAINING TIMES")
    
    plt.figure(figsize=(12, 6))
    
    # Sort by time
    sorted_times = dict(sorted(execution_times.items(), key=lambda x: x[1]))
    
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(sorted_times)))
    bars = plt.bar(range(len(sorted_times)), list(sorted_times.values()), color=colors, edgecolor='black')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Model Training Time Comparison', fontsize=16, fontweight='bold')
    plt.xticks(range(len(sorted_times)), list(sorted_times.keys()), rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results_training_times.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved results_training_times.png")
    # plt.show()

# ==================== 6. MAIN EXECUTION ====================

def main():
    """Main execution function"""
    import time
    
    print("\n🚀 STARTING COMPLETE ML PROJECT WITH VISUALIZATIONS")
    print("=" * 100)
    
    # Track execution times
    execution_times = {}
    
    # Step 1: Create and visualize dataset
    start_time = time.time()
    df = create_and_visualize_dataset()
    execution_times['Dataset Creation'] = time.time() - start_time
    
    # Step 2: Preprocess data
    start_time = time.time()
    X, y, feature_names, df_processed = preprocess_data(df)
    execution_times['Data Preprocessing'] = time.time() - start_time
    
    # Step 3: Split data
    print("\n📊 SPLITTING DATA")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"✅ Training samples: {X_train.shape[0]:,}")
    print(f"✅ Testing samples: {X_test.shape[0]:,}")
    
    # Step 4: Train all models
    start_time = time.time()
    results_df, models, predictions, prob_predictions = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, feature_names
    )
    execution_times['Model Training'] = time.time() - start_time
    
    # Step 5: Display results table
    print("\n" + "=" * 100)
    print("RESULTS TABLE")
    print("=" * 100)
    
    # Format and display table
    display_df = results_df.copy()
    display_df['Accuracy'] = display_df['Accuracy'].map('{:.4f}'.format)
    display_df['Precision'] = display_df['Precision'].map('{:.4f}'.format)
    display_df['Recall'] = display_df['Recall'].map('{:.4f}'.format)
    display_df['F1-Score'] = display_df['F1-Score'].map('{:.4f}'.format)
    
    print(display_df.to_string(index=False))
    
    # Step 6: Create all visualizations
    start_time = time.time()
    
    # 6.1 Model comparison bar charts
    plot_model_comparison(results_df)
    
    # 6.2 Confusion matrices
    plot_confusion_matrices(predictions, y_test, results_df['Model'].tolist())
    
    # 6.3 ROC curves (for models with probability predictions)
    plot_roc_curves(prob_predictions, y_test, results_df['Model'].tolist())
    
    # 6.4 Precision-Recall curves
    plot_precision_recall_curves(prob_predictions, y_test, results_df['Model'].tolist())
    
    # 6.5 Feature importance
    plot_feature_importance(models, feature_names)
    
    # 6.6 Decision tree visualization
    plot_decision_tree(models, feature_names)
    
    execution_times['Visualizations'] = time.time() - start_time
    
    # Step 7: Plot execution times
    plot_model_training_time(results_df, execution_times)
    
    # Step 8: Find and display best model
    best_model_idx = results_df['Accuracy'].idxmax()
    best_model = results_df.loc[best_model_idx]
    
    print("\n" + "=" * 100)
    print("🏆 BEST MODEL ANALYSIS")
    print("=" * 100)
    
    # Create best model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [best_model[metric] for metric in metrics]
    
    # Calculate average of other models
    other_models_avg = results_df[results_df['Model'] != best_model['Model']][metrics].mean().values
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, values, width, label=best_model['Model'], color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, other_models_avg, width, label='Other Models Average', color='#e74c3c', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Best Model: {best_model["Model"]} vs Other Models Average', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results_best_model_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved results_best_model_comparison.png")
    # plt.show()
    
    # Display best model info
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model['Model']}")
    print(f"   📚 Library: {best_model['Library']}")
    print(f"   🎯 Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   🎯 Precision: {best_model['Precision']:.4f}")
    print(f"   🎯 Recall: {best_model['Recall']:.4f}")
    print(f"   ⚡ F1-Score: {best_model['F1-Score']:.4f}")
    
    # Step 9: Save results
    print("\n" + "=" * 100)
    print("💾 SAVING RESULTS")
    print("=" * 100)
    
    # Save results to CSV
    results_df.to_csv('ml_project_results.csv', index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df['Actual'] = y_test
    predictions_df.to_csv('model_predictions.csv', index=False)
    
    print("✅ Results saved to 'ml_project_results.csv'")
    print("✅ Predictions saved to 'model_predictions.csv'")
    
    # Step 10: Final summary
    print("\n" + "=" * 100)
    print("📋 PROJECT SUMMARY")
    print("=" * 100)
    
    print(f"\n✅ PROJECT COMPLETED SUCCESSFULLY!")
    print(f"\n📊 MODELS IMPLEMENTED: {len(results_df)}")
    print(f"📈 VISUALIZATIONS CREATED: 8 different types")
    print(f"📁 FILES GENERATED: 2 CSV files")
    
    print(f"\n⏱️ EXECUTION TIMES:")
    for stage, time_taken in execution_times.items():
        print(f"   {stage}: {time_taken:.2f} seconds")
    
    print(f"\n🎯 BEST MODEL: {best_model['Model']}")
    print(f"   📊 Accuracy: {best_model['Accuracy']:.4f}")
    
    return results_df, models, predictions

# ==================== 7. RUN THE PROJECT ====================

if __name__ == "__main__":
    # Check for required packages
    # Check for required packages
    required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn']
    
    # Optional packages
    optional_packages = ['xgboost', 'lightgbm', 'catboost']
    
    # CatBoost is optional
    optional_packages = ['catboost']
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Please install using:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("✅ All required packages are installed!")
        print("Starting project execution...\n")
        
        try:
            results, models, predictions = main()
            
            print("\n" + "=" * 100)
            print("🎉 PROJECT EXECUTION COMPLETE! 🎉")
            print("=" * 100)
            print("\n📊 Check the generated CSV files for detailed results.")
            print("🖼️ All visualizations have been displayed.")
            print("\n📁 Files created:")
            print("   - ml_project_results.csv (All metrics)")
            print("   - model_predictions.csv (Predictions from all models)")
            print("\n✅ You can now use these results for your project report.")
            
        except Exception as e:
            print(f"\n❌ Error during execution: {str(e)}")
            import traceback
            traceback.print_exc()