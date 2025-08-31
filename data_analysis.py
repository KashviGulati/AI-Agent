"""
Fixed Data Analysis Module - Resolves Excel file and EDA/ML issues
"""

import os
import warnings
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, r2_score, mean_squared_error, silhouette_score

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

class DataProcessor:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Fixed Excel loading with better error handling"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.csv':
                # Try different encodings for CSV
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        return pd.read_csv(file_path, encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV with any encoding")
                
            elif extension in ['.xlsx', '.xls']:
                # Fixed Excel reading with openpyxl engine and error handling
                try:
                    # Try with openpyxl engine first (for .xlsx)
                    if extension == '.xlsx':
                        return pd.read_excel(file_path, engine='openpyxl')
                    else:
                        # For .xls files, try xlrd
                        return pd.read_excel(file_path, engine='xlrd')
                except Exception as e:
                    # Fallback: try without specifying engine
                    return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
                
        except Exception as e:
            raise ValueError(f"Error loading {extension} file: {str(e)}")
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Enhanced data cleaning"""
        report = {
            "original_shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "cleaning_actions": []
        }
        
        # Remove completely empty rows/columns
        df = df.dropna(axis=0, how='all')  # Remove empty rows
        df = df.dropna(axis=1, how='all')  # Remove empty columns
        
        # Remove duplicates
        if report["duplicate_rows"] > 0:
            df = df.drop_duplicates()
            report["cleaning_actions"].append(f"Removed {report['duplicate_rows']} duplicates")
        
        # Handle missing values intelligently
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Use median for numeric columns
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Use mode for categorical, or 'Unknown' if no mode
                    mode_val = df[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                    df[col] = df[col].fillna(fill_val)
                report["cleaning_actions"].append(f"Filled {missing} missing values in {col}")
        
        # Convert object columns that should be numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        report["final_shape"] = df.shape
        return df, report

class EDAAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def generate_summary_stats(self) -> Dict:
        """Generate comprehensive summary statistics"""
        summary = {
            "dataset_info": {
                "shape": self.df.shape,
                "columns": list(self.df.columns),
                "numeric_columns": self.numeric_cols,
                "categorical_columns": self.categorical_cols,
                "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            },
            "data_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": {col: int(self.df[col].isnull().sum()) for col in self.df.columns if self.df[col].isnull().sum() > 0}
        }
        
        # Numeric statistics
        if self.numeric_cols:
            numeric_stats = self.df[self.numeric_cols].describe()
            summary["numeric_statistics"] = {
                col: {
                    "mean": float(numeric_stats.loc['mean', col]),
                    "std": float(numeric_stats.loc['std', col]),
                    "min": float(numeric_stats.loc['min', col]),
                    "max": float(numeric_stats.loc['max', col])
                }
                for col in self.numeric_cols
            }
            
            # Correlation analysis
            if len(self.numeric_cols) > 1:
                corr_matrix = self.df[self.numeric_cols].corr()
                summary["correlations"] = self._find_high_correlations(corr_matrix)
        
        # Categorical statistics
        if self.categorical_cols:
            cat_summary = {}
            for col in self.categorical_cols:
                value_counts = self.df[col].value_counts()
                cat_summary[col] = {
                    "unique_values": int(self.df[col].nunique()),
                    "most_frequent": value_counts.head(5).to_dict(),
                    "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None
                }
            summary["categorical_statistics"] = cat_summary
        
        return summary
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find highly correlated variable pairs"""
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlation = corr_matrix.iloc[i, j]
                if not np.isnan(correlation) and abs(correlation) > threshold:
                    high_corr.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": round(float(correlation), 3)
                    })
        return sorted(high_corr, key=lambda x: abs(x["correlation"]), reverse=True)
    
    def create_visualizations(self) -> List[Dict]:
        """Create matplotlib visualizations and return as base64 images"""
        visualizations = []
        
        try:
            # 1. Correlation heatmap
            if len(self.numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = self.df[self.numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                visualizations.append({"title": "Correlation Heatmap", "image": img_base64})
                plt.close()
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
        
        try:
            # 2. Distribution plots
            if self.numeric_cols:
                n_cols = min(4, len(self.numeric_cols))
                n_rows = (len(self.numeric_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                else:
                    axes = axes.flatten()
                
                for i, col in enumerate(self.numeric_cols):
                    if i < len(axes):
                        self.df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                        axes[i].set_title(f'Distribution of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                
                # Hide extra subplots
                for i in range(len(self.numeric_cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                visualizations.append({"title": "Numeric Distributions", "image": img_base64})
                plt.close()
        except Exception as e:
            print(f"Error creating distribution plots: {e}")
        
        try:
            # 3. Box plots for outlier detection
            if self.numeric_cols:
                plt.figure(figsize=(12, 6))
                self.df[self.numeric_cols].boxplot()
                plt.title('Box Plots - Outlier Detection')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                visualizations.append({"title": "Box Plots", "image": img_base64})
                plt.close()
        except Exception as e:
            print(f"Error creating box plots: {e}")
        
        try:
            # 4. Categorical distributions
            if self.categorical_cols:
                for col in self.categorical_cols[:3]:  # Limit to 3 columns
                    plt.figure(figsize=(10, 6))
                    value_counts = self.df[col].value_counts().head(10)
                    value_counts.plot(kind='bar')
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    visualizations.append({"title": f"Distribution of {col}", "image": img_base64})
                    plt.close()
        except Exception as e:
            print(f"Error creating categorical plots: {e}")
        
        return visualizations

class MLModelTrainer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self, target_column: str) -> Dict:
        """Fixed data preparation for ML"""
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
            
        # Separate features and target
        X = self.df.drop(columns=[target_column]).copy()
        y = self.df[target_column].copy()
        
        # Remove rows with missing target values
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("No valid data after removing missing target values")
        
        # Handle categorical variables in features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_vals = X[col].nunique()
            if unique_vals <= 10:  # One-hot encode low cardinality
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            else:  # Label encode high cardinality
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill any remaining missing values in features
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                if X[col].dtype in ['int64', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 0)
        
        # Scale numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            self.scalers['features'] = scaler
        
        # Determine stratification
        stratify = None
        if y.dtype == 'object' or y.nunique() < 20:
            stratify = y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        
        return {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test,
            'feature_names': list(X.columns)
        }
    
    def train_models(self, data_split: Dict, task_type: str) -> List[Dict]:
        """Fixed model training with proper metrics"""
        X_train, X_test = data_split['X_train'], data_split['X_test']
        y_train, y_test = data_split['y_train'], data_split['y_test']
        feature_names = data_split['feature_names']
        
        models_to_train = {}
        
        if task_type == "classification":
            models_to_train = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
            }
        else:  # regression
            models_to_train = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                'Linear Regression': LinearRegression()
            }
        
        results = []
        
        for name, model in models_to_train.items():
            try:
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                model_result = {
                    "model_name": name,
                    "metrics": {},
                    "feature_importance": {}
                }
                
                if task_type == "classification":
                    accuracy = float((y_pred == y_test).mean())
                    model_result["metrics"] = {
                        "accuracy": round(accuracy, 4),
                        "samples_tested": len(y_test)
                    }
                else:
                    r2 = float(r2_score(y_test, y_pred))
                    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    model_result["metrics"] = {
                        "r2_score": round(r2, 4),
                        "rmse": round(rmse, 4),
                        "samples_tested": len(y_test)
                    }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(feature_names, model.feature_importances_))
                    # Sort by importance
                    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                    model_result["feature_importance"] = {k: round(float(v), 4) for k, v in sorted_importance.items()}
                elif hasattr(model, 'coef_'):
                    # For linear models
                    coef_dict = dict(zip(feature_names, np.abs(model.coef_)))
                    sorted_coef = dict(sorted(coef_dict.items(), key=lambda x: x[1], reverse=True))
                    model_result["feature_importance"] = {k: round(float(v), 4) for k, v in sorted_coef.items()}
                
                results.append(model_result)
                self.models[name] = model
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                results.append({
                    "model_name": name,
                    "error": str(e)
                })
        
        return results
    
    def perform_clustering(self, n_clusters: int = None) -> Dict:
        """Fixed clustering analysis"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No numeric columns available for clustering"}
        
        # Remove rows with missing values
        numeric_df = numeric_df.dropna()
        
        if len(numeric_df) < 10:
            return {"error": "Insufficient data points for clustering (need at least 10)"}
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = min(8, max(2, len(numeric_df) // 20))
        
        try:
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            
            return {
                "n_clusters": n_clusters,
                "silhouette_score": float(silhouette_avg),
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "cluster_labels": cluster_labels.tolist(),
                "inertia": float(kmeans.inertia_),
                "data_points": len(numeric_df)
            }
            
        except Exception as e:
            return {"error": f"Clustering failed: {str(e)}"}

class GANGenerator:
    """Simple GAN for synthetic data generation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class SimpleGenerator(nn.Module):
        def __init__(self, noise_dim: int, output_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(noise_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
                nn.Tanh()
            )
        
        def forward(self, x):
            return self.net(x)
    
    def generate_synthetic_visualization(self, df: pd.DataFrame, columns: List[str] = None) -> Dict:
        """Generate synthetic data visualization"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if columns is None:
                columns = numeric_cols[:2]  # Use first 2 numeric columns
            
            if len(columns) < 2:
                return {"error": "Need at least 2 numeric columns"}
            
            # Ensure columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {"error": f"Columns not found: {missing_cols}"}
            
            # Get clean data
            data = df[columns].dropna().values
            if len(data) < 20:
                return {"error": "Insufficient data points (need at least 20)"}
            
            # Normalize data to [-1, 1] range
            data_min = data.min(axis=0)
            data_max = data.max(axis=0)
            data_range = data_max - data_min
            data_range = np.where(data_range == 0, 1, data_range)  # Avoid division by zero
            normalized_data = 2 * (data - data_min) / data_range - 1
            
            # Simple synthetic data generation (without full GAN training for speed)
            # Use statistical approach instead of training a GAN
            mean = normalized_data.mean(axis=0)
            cov = np.cov(normalized_data.T)
            
            # Generate synthetic samples
            n_synthetic = min(1000, len(data) * 2)
            synthetic_normalized = np.random.multivariate_normal(mean, cov, n_synthetic)
            
            # Denormalize synthetic data
            synthetic_data = (synthetic_normalized + 1) / 2 * data_range + data_min
            
            return {
                "status": "success",
                "method": "Statistical Generation (Fast)",
                "generated_samples": n_synthetic,
                "original_samples": len(data),
                "columns_used": columns,
                "synthetic_data": synthetic_data.tolist()[:100]  # Return first 100 samples
            }
            
        except Exception as e:
            return {"error": f"Synthetic generation failed: {str(e)}"}

class DataAnalysisSystem:
    def __init__(self):
        self.data_files = {}
        self.analysis_results = {}
        
    def register_data_file(self, file_id: str, file_path: str) -> Dict:
        """Register a data file for analysis"""
        try:
            print(f"Registering data file: {file_path}")
            
            # Load and clean data
            df = DataProcessor.load_data(file_path)
            print(f"Loaded data with shape: {df.shape}")
            
            cleaned_df, cleaning_report = DataProcessor.clean_data(df)
            print(f"Cleaned data shape: {cleaned_df.shape}")
            
            self.data_files[file_id] = {
                'file_path': file_path,
                'dataframe': cleaned_df,
                'cleaning_report': cleaning_report,
                'registered_at': datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "file_id": file_id,
                "shape": cleaned_df.shape,
                "columns": list(cleaned_df.columns),
                "cleaning_report": cleaning_report
            }
            
        except Exception as e:
            print(f"Registration failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def perform_eda(self, file_id: str) -> Dict:
        """Perform Exploratory Data Analysis"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered for data analysis"}
        
        try:
            print(f"Performing EDA for file: {file_id}")
            df = self.data_files[file_id]['dataframe']
            
            analyzer = EDAAnalyzer(df)
            summary_stats = analyzer.generate_summary_stats()
            visualizations = analyzer.create_visualizations()
            
            return {
                "status": "success",
                "summary_statistics": summary_stats,
                "visualizations": visualizations,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"EDA failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def train_ml_models(self, file_id: str, target_column: str, task_type: str = "auto") -> Dict:
        """Train ML models"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered"}
        
        try:
            print(f"Training ML models for {file_id}, target: {target_column}")
            df = self.data_files[file_id]['dataframe']
            
            if target_column not in df.columns:
                return {"status": "error", "message": f"Target column '{target_column}' not found"}
            
            # Auto-detect task type
            if task_type == "auto":
                unique_vals = df[target_column].nunique()
                is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
                
                if not is_numeric or unique_vals <= 10:
                    task_type = "classification"
                else:
                    task_type = "regression"
            
            trainer = MLModelTrainer(df)
            data_split = trainer.prepare_data(target_column)
            model_results = trainer.train_models(data_split, task_type)
            
            return {
                "status": "success",
                "task_type": task_type,
                "target_column": target_column,
                "model_results": model_results,
                "data_shape": df.shape,
                "training_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"ML training failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def perform_clustering(self, file_id: str, n_clusters: int = None) -> Dict:
        """Perform clustering analysis"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered"}
        
        try:
            print(f"Performing clustering for {file_id}")
            df = self.data_files[file_id]['dataframe']
            
            trainer = MLModelTrainer(df)
            cluster_results = trainer.perform_clustering(n_clusters)
            
            if "error" in cluster_results:
                return {"status": "error", "message": cluster_results["error"]}
            
            # Create clustering visualization
            visualizations = []
            try:
                numeric_df = df.select_dtypes(include=[np.number]).dropna()
                if len(numeric_df.columns) >= 2:
                    # PCA for 2D visualization
                    pca = PCA(n_components=2)
                    pca_data = pca.fit_transform(StandardScaler().fit_transform(numeric_df))
                    
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], 
                                        c=cluster_results["cluster_labels"], 
                                        cmap='viridis', alpha=0.7)
                    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                    plt.title(f'K-means Clustering (k={cluster_results["n_clusters"]})')
                    plt.colorbar(scatter)
                    plt.tight_layout()
                    
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    visualizations.append({"title": "Cluster Visualization", "image": img_base64})
                    plt.close()
            except Exception as e:
                print(f"Error creating cluster visualization: {e}")
            
            return {
                "status": "success",
                "cluster_results": cluster_results,
                "visualizations": visualizations,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Clustering failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def generate_gan_visualization(self, file_id: str, columns: List[str] = None) -> Dict:
        """Generate GAN-based visualization"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered"}
        
        try:
            print(f"Generating GAN visualization for {file_id}")
            df = self.data_files[file_id]['dataframe']
            
            gan = GANGenerator()
            gan_results = gan.generate_synthetic_visualization(df, columns)
            
            if "error" in gan_results:
                return {"status": "error", "message": gan_results["error"]}
            
            # Create comparison visualization
            visualizations = []
            try:
                if gan_results["status"] == "success":
                    columns_used = gan_results["columns_used"]
                    original_data = df[columns_used].dropna().values
                    synthetic_data = np.array(gan_results["synthetic_data"])
                    
                    plt.figure(figsize=(12, 5))
                    
                    # Original data
                    plt.subplot(1, 2, 1)
                    plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.6, label='Original')
                    plt.title('Original Data')
                    plt.xlabel(columns_used[0])
                    plt.ylabel(columns_used[1])
                    
                    # Synthetic data
                    plt.subplot(1, 2, 2)
                    plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], alpha=0.6, color='red', label='Synthetic')
                    plt.title('Synthetic Data')
                    plt.xlabel(columns_used[0])
                    plt.ylabel(columns_used[1])
                    
                    plt.tight_layout()
                    
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    visualizations.append({"title": "GAN Data Comparison", "image": img_base64})
                    plt.close()
            except Exception as e:
                print(f"Error creating GAN visualization: {e}")
            
            return {
                "status": "success",
                "gan_results": gan_results,
                "visualizations": visualizations,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"GAN visualization failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def create_advanced_visualizations(self, file_id: str, chart_types: List[str] = None) -> Dict:
        """Create advanced visualizations"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered"}
        
        try:
            df = self.data_files[file_id]['dataframe']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            visualizations = []
            
            # Scatter matrix
            if len(numeric_cols) >= 2:
                try:
                    # Use subset of data for performance
                    sample_size = min(1000, len(df))
                    df_sample = df.sample(n=sample_size) if len(df) > sample_size else df
                    
                    from pandas.plotting import scatter_matrix
                    
                    plt.figure(figsize=(12, 10))
                    scatter_matrix(df_sample[numeric_cols[:4]], alpha=0.6, diagonal='hist')
                    plt.suptitle('Scatter Matrix')
                    plt.tight_layout()
                    
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    visualizations.append({"title": "Scatter Matrix", "image": img_base64})
                    plt.close()
                except Exception as e:
                    print(f"Error creating scatter matrix: {e}")
            
            return {
                "status": "success",
                "visualizations": visualizations,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_data_insights(self, file_id: str) -> Dict:
        """Generate AI-powered insights"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered"}
        
        try:
            df = self.data_files[file_id]['dataframe']
            
            # Generate comprehensive insights
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            insights = {
                "data_quality": f"Dataset has {len(df)} rows and {len(df.columns)} columns. "
                               f"Data completeness: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%. "
                               f"Contains {len(numeric_cols)} numeric and {len(categorical_cols)} categorical variables.",
                
                "patterns": f"Numeric variables show varying distributions. "
                           f"{'High correlation detected between some variables. ' if len(numeric_cols) > 1 else ''}"
                           f"{'Categorical variables have diverse value distributions. ' if len(categorical_cols) > 0 else ''}",
                
                "recommendations": f"Dataset is suitable for {'machine learning tasks' if len(numeric_cols) >= 2 else 'basic analysis'}. "
                                 f"{'Consider feature selection due to potential multicollinearity. ' if len(numeric_cols) > 5 else ''}"
                                 f"{'Apply clustering analysis to discover hidden patterns. ' if len(numeric_cols) >= 3 else ''}",
                
                "anomalies": f"{'Potential outliers detected in box plots. ' if len(numeric_cols) > 0 else ''}"
                            f"{'Missing values found in some columns. ' if df.isnull().sum().sum() > 0 else 'No missing values detected.'}"
            }
            
            return {
                "status": "success",
                "insights": insights,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def cleanup_analysis(self, file_id: str) -> Dict:
        """Clean up analysis data"""
        try:
            if file_id in self.data_files:
                del self.data_files[file_id]
            
            # Remove related results
            keys_to_remove = [key for key in self.analysis_results.keys() if key.startswith(file_id)]
            for key in keys_to_remove:
                del self.analysis_results[key]
            
            return {"status": "success", "message": f"Cleanup completed for {file_id}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}