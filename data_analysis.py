"""
data_analysis.py - Advanced Data Analysis Module
Handles EDA, ML models, GANs, and data visualization for CSV/Excel files
"""

import os
import io
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import base64

# Data processing and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

# Deep Learning for GANs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr

# Suppress warnings
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

class DataProcessor:
    """Enhanced data processing for CSV/Excel files"""
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load CSV or Excel file into DataFrame"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"Successfully loaded CSV with {encoding} encoding")
                        return df
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV file with any encoding")
                
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                print(f"Successfully loaded Excel file")
                return df
                
            else:
                raise ValueError(f"Unsupported file format: {extension}")
                
        except Exception as e:
            raise ValueError(f"Error loading data file: {str(e)}")
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Clean and preprocess data"""
        cleaning_report = {
            "original_shape": df.shape,
            "missing_values": {},
            "duplicate_rows": 0,
            "data_types": {},
            "cleaning_actions": []
        }
        
        # Check for missing values
        missing_values = df.isnull().sum()
        cleaning_report["missing_values"] = missing_values.to_dict()
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        cleaning_report["duplicate_rows"] = int(duplicates)
        
        # Data types
        cleaning_report["data_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Remove duplicates
        if duplicates > 0:
            df = df.drop_duplicates()
            cleaning_report["cleaning_actions"].append(f"Removed {duplicates} duplicate rows")
        
        # Handle missing values (basic strategy)
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                if df[column].dtype in ['int64', 'float64']:
                    df[column] = df[column].fillna(df[column].median())
                    cleaning_report["cleaning_actions"].append(f"Filled {missing_count} missing values in '{column}' with median")
                else:
                    df[column] = df[column].fillna(df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown')
                    cleaning_report["cleaning_actions"].append(f"Filled {missing_count} missing values in '{column}' with mode")
        
        cleaning_report["final_shape"] = df.shape
        
        return df, cleaning_report

class EDAAnalyzer:
    """Comprehensive Exploratory Data Analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def generate_summary_stats(self) -> Dict:
        """Generate comprehensive summary statistics"""
        summary = {
            "dataset_info": {
                "shape": self.df.shape,
                "columns": list(self.df.columns),
                "numeric_columns": self.numeric_cols,
                "categorical_columns": self.categorical_cols,
                "datetime_columns": self.datetime_cols,
                "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            "summary_statistics": {},
            "correlation_analysis": {},
            "outlier_analysis": {},
            "distribution_analysis": {}
        }
        
        # Summary statistics for numeric columns
        if self.numeric_cols:
            summary["summary_statistics"]["numeric"] = self.df[self.numeric_cols].describe().to_dict()
            
            # Correlation analysis
            corr_matrix = self.df[self.numeric_cols].corr()
            summary["correlation_analysis"] = {
                "correlation_matrix": corr_matrix.to_dict(),
                "high_correlations": self._find_high_correlations(corr_matrix)
            }
            
            # Outlier analysis using IQR method
            outliers = {}
            for col in self.numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                outliers[col] = {
                    "count": int(outlier_count),
                    "percentage": round(outlier_count / len(self.df) * 100, 2),
                    "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                }
            summary["outlier_analysis"] = outliers
        
        # Categorical analysis
        if self.categorical_cols:
            categorical_summary = {}
            for col in self.categorical_cols:
                value_counts = self.df[col].value_counts().head(10)
                categorical_summary[col] = {
                    "unique_values": int(self.df[col].nunique()),
                    "most_frequent": value_counts.to_dict(),
                    "missing_percentage": round(self.df[col].isnull().sum() / len(self.df) * 100, 2)
                }
            summary["summary_statistics"]["categorical"] = categorical_summary
        
        return summary
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find highly correlated variable pairs"""
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) > threshold:
                    high_corr_pairs.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": round(correlation, 3)
                    })
        return sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)
    
    def create_visualizations(self) -> Dict[str, str]:
        """Create comprehensive EDA visualizations"""
        visualizations = {}
        
        # 1. Correlation heatmap
        if len(self.numeric_cols) > 1:
            fig = px.imshow(
                self.df[self.numeric_cols].corr(),
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale="RdBu"
            )
            visualizations["correlation_heatmap"] = fig.to_json()
        
        # 2. Distribution plots for numeric variables
        if self.numeric_cols:
            fig = make_subplots(
                rows=(len(self.numeric_cols) + 2) // 3,
                cols=3,
                subplot_titles=self.numeric_cols,
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(self.numeric_cols):
                row = (i // 3) + 1
                col_pos = (i % 3) + 1
                
                fig.add_trace(
                    go.Histogram(x=self.df[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(title_text="Distribution of Numeric Variables", height=300 * ((len(self.numeric_cols) + 2) // 3))
            visualizations["numeric_distributions"] = fig.to_json()
        
        # 3. Box plots for outlier detection
        if self.numeric_cols:
            fig = go.Figure()
            for col in self.numeric_cols:
                fig.add_trace(go.Box(y=self.df[col], name=col))
            fig.update_layout(title="Box Plots for Outlier Detection")
            visualizations["box_plots"] = fig.to_json()
        
        # 4. Categorical variable analysis
        if self.categorical_cols:
            # Take first few categorical columns to avoid overcrowding
            cols_to_plot = self.categorical_cols[:4]
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=cols_to_plot,
                specs=[[{"type": "xy"}, {"type": "xy"}], 
                       [{"type": "xy"}, {"type": "xy"}]]
            )
            
            for i, col in enumerate(cols_to_plot):
                row = (i // 2) + 1
                col_pos = (i % 2) + 1
                
                value_counts = self.df[col].value_counts().head(10)
                fig.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values, name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(title_text="Categorical Variable Distributions", height=600)
            visualizations["categorical_distributions"] = fig.to_json()
        
        return visualizations

class MLModelTrainer:
    """Machine Learning Model Training and Evaluation"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def prepare_data(self, target_column: str, test_size: float = 0.2) -> Dict:
        """Prepare data for machine learning"""
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Separate features and target
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Handle categorical variables
        X_processed = X.copy()
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if X_processed[col].nunique() <= 10:  # One-hot encode if few categories
                dummies = pd.get_dummies(X_processed[col], prefix=col)
                X_processed = pd.concat([X_processed.drop(columns=[col]), dummies], axis=1)
            else:  # Label encode if many categories
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.encoders[col] = le
        
        # Scale numeric features
        scaler = StandardScaler()
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
        self.scalers['features'] = scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42
        )
        
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'feature_names': list(X_processed.columns)
        }
    
    def train_classification_models(self, data_split: Dict, target_column: str) -> Dict:
        """Train multiple classification models"""
        X_train, X_test = data_split['X_train'], data_split['X_test']
        y_train, y_test = data_split['y_train'], data_split['y_test']
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = (y_pred_train == y_train).mean()
            test_accuracy = (y_pred_test == y_test).mean()
            
            # Classification report
            class_report = classification_report(y_test, y_pred_test, output_dict=True)
            
            results[name] = {
                'model': model,
                'train_accuracy': round(train_accuracy, 4),
                'test_accuracy': round(test_accuracy, 4),
                'classification_report': class_report,
                'feature_importance': self._get_feature_importance(model, data_split['feature_names'])
            }
            
            self.models[f"{name}_{target_column}"] = model
        
        return results
    
    def train_regression_models(self, data_split: Dict, target_column: str) -> Dict:
        """Train multiple regression models"""
        X_train, X_test = data_split['X_train'], data_split['X_test']
        y_train, y_test = data_split['y_train'], data_split['y_test']
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            results[name] = {
                'model': model,
                'train_r2': round(train_r2, 4),
                'test_r2': round(test_r2, 4),
                'train_rmse': round(train_rmse, 4),
                'test_rmse': round(test_rmse, 4),
                'feature_importance': self._get_feature_importance(model, data_split['feature_names'])
            }
            
            self.models[f"{name}_{target_column}"] = model
        
        return results
    
    def perform_clustering(self, n_clusters: int = None) -> Dict:
        """Perform K-means clustering analysis"""
        # Select only numeric columns for clustering
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No numeric columns found for clustering"}
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            inertias = []
            k_range = range(1, min(11, len(numeric_df) // 2))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
            
            # Use elbow method (simplified)
            n_clusters = 3  # Default fallback
            if len(inertias) > 2:
                # Find the elbow point
                diffs = np.diff(inertias)
                elbow_index = np.argmax(diffs[:-1] - diffs[1:]) + 2
                n_clusters = min(elbow_index, 5)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to dataframe
        clustered_df = numeric_df.copy()
        clustered_df['Cluster'] = cluster_labels
        
        # Cluster analysis
        cluster_summary = clustered_df.groupby('Cluster').agg(['mean', 'std']).round(3)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        # Create visualization
        fig = px.scatter(
            x=pca_data[:, 0], y=pca_data[:, 1],
            color=cluster_labels,
            title=f"K-means Clustering (k={n_clusters}) - PCA Visualization",
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'}
        )
        
        results = {
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'cluster_summary': cluster_summary.to_dict(),
            'pca_visualization': fig.to_json(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        }
        
        self.models[f'kmeans_{n_clusters}'] = kmeans
        self.scalers['clustering'] = scaler
        
        return results
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """Extract feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance))
            # Sort by importance
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(model, 'coef_'):
            # For linear models
            importance = np.abs(model.coef_).flatten()
            feature_importance = dict(zip(feature_names, importance))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}

class GANDiagramGenerator:
    """GAN-based diagram and data visualization generator"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = None
        self.discriminator = None
        
    class SimpleGenerator(nn.Module):
        """Simple generator for creating synthetic data patterns"""
        def __init__(self, noise_dim: int, output_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(noise_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
                nn.Tanh()
            )
        
        def forward(self, x):
            return self.net(x)
    
    class SimpleDiscriminator(nn.Module):
        """Simple discriminator for GAN training"""
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    def generate_synthetic_data_visualization(self, df: pd.DataFrame, columns: List[str] = None) -> Dict:
        """Generate synthetic data patterns using a simple GAN approach"""
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols[:2]  # Take first 2 numeric columns
        
        if len(columns) < 2:
            return {"error": "Need at least 2 numeric columns for GAN visualization"}
        
        # Prepare data
        data = df[columns].dropna().values
        data = (data - data.mean(axis=0)) / data.std(axis=0)  # Normalize
        
        # Simple approach: generate synthetic data points that follow similar distribution
        noise_dim = 10
        output_dim = len(columns)
        
        # Initialize models
        generator = self.SimpleGenerator(noise_dim, output_dim).to(self.device)
        discriminator = self.SimpleDiscriminator(output_dim).to(self.device)
        
        # Training parameters
        lr = 0.0002
        epochs = 100
        batch_size = min(32, len(data) // 2)
        
        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        
        # Create data loader
        tensor_data = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop (simplified)
        for epoch in range(epochs):
            for batch_data in dataloader:
                real_data = batch_data[0]
                batch_size_current = real_data.size(0)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                
                # Real data
                real_labels = torch.ones(batch_size_current, 1).to(self.device)
                real_output = discriminator(real_data)
                real_loss = criterion(real_output, real_labels)
                
                # Fake data
                noise = torch.randn(batch_size_current, noise_dim).to(self.device)
                fake_data = generator(noise)
                fake_labels = torch.zeros(batch_size_current, 1).to(self.device)
                fake_output = discriminator(fake_data.detach())
                fake_loss = criterion(fake_output, fake_labels)
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                fake_output = discriminator(fake_data)
                g_loss = criterion(fake_output, real_labels)
                g_loss.backward()
                optimizer_G.step()
        
        # Generate synthetic data for visualization
        with torch.no_grad():
            noise = torch.randn(500, noise_dim).to(self.device)
            synthetic_data = generator(noise).cpu().numpy()
        
        # Create visualization
        fig = go.Figure()
        
        # Plot real data
        fig.add_trace(go.Scatter(
            x=data[:, 0], y=data[:, 1],
            mode='markers',
            name='Real Data',
            marker=dict(color='blue', size=5, opacity=0.6)
        ))
        
        # Plot synthetic data
        fig.add_trace(go.Scatter(
            x=synthetic_data[:, 0], y=synthetic_data[:, 1],
            mode='markers',
            name='GAN Generated Data',
            marker=dict(color='red', size=5, opacity=0.6)
        ))
        
        fig.update_layout(
            title='GAN-Generated Data vs Real Data',
            xaxis_title=columns[0],
            yaxis_title=columns[1],
            showlegend=True
        )
        
        return {
            'visualization': fig.to_json(),
            'real_data_points': len(data),
            'synthetic_data_points': len(synthetic_data),
            'columns_used': columns,
            'training_epochs': epochs
        }

class DataAnalysisSystem:
    """Main system for comprehensive data analysis"""
    
    def __init__(self):
        self.data_files = {}  # Store loaded data files
        self.analysis_results = {}  # Store analysis results
        
    def register_data_file(self, file_id: str, file_path: str) -> Dict:
        """Register a CSV/Excel file for data analysis"""
        try:
            # Check if file is CSV or Excel
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in ['.csv', '.xlsx', '.xls']:
                return {"status": "error", "message": f"Unsupported file type: {file_ext}"}
            
            # Load and clean data
            df = DataProcessor.load_data(file_path)
            cleaned_df, cleaning_report = DataProcessor.clean_data(df)
            
            self.data_files[file_id] = {
                'file_path': file_path,
                'dataframe': cleaned_df,
                'original_dataframe': df,
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
            return {"status": "error", "message": str(e)}
    
    def perform_eda(self, file_id: str) -> Dict:
        """Perform comprehensive EDA on registered data file"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered for data analysis"}
        
        try:
            df = self.data_files[file_id]['dataframe']
            eda_analyzer = EDAAnalyzer(df)
            
            # Generate summary statistics
            summary_stats = eda_analyzer.generate_summary_stats()
            
            # Create visualizations
            visualizations = eda_analyzer.create_visualizations()
            
            results = {
                "status": "success",
                "file_id": file_id,
                "summary_statistics": summary_stats,
                "visualizations": visualizations,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.analysis_results[f"{file_id}_eda"] = results
            return results
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def train_ml_models(self, file_id: str, target_column: str, task_type: str = "auto") -> Dict:
        """Train ML models on the dataset"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered for data analysis"}
        
        try:
            df = self.data_files[file_id]['dataframe']
            ml_trainer = MLModelTrainer(df)
            
            # Prepare data
            data_split = ml_trainer.prepare_data(target_column)
            
            # Determine task type automatically if not specified
            if task_type == "auto":
                target_nunique = df[target_column].nunique()
                if target_nunique <= 10 and df[target_column].dtype == 'object':
                    task_type = "classification"
                elif df[target_column].dtype in ['int64', 'float64'] and target_nunique > 10:
                    task_type = "regression"
                else:
                    task_type = "classification"
            
            # Train models based on task type
            if task_type == "classification":
                model_results = ml_trainer.train_classification_models(data_split, target_column)
            elif task_type == "regression":
                model_results = ml_trainer.train_regression_models(data_split, target_column)
            else:
                return {"status": "error", "message": f"Unsupported task type: {task_type}"}
            
            results = {
                "status": "success",
                "file_id": file_id,
                "target_column": target_column,
                "task_type": task_type,
                "model_results": model_results,
                "data_split_info": {
                    "train_samples": len(data_split['X_train']),
                    "test_samples": len(data_split['X_test']),
                    "features_count": len(data_split['feature_names'])
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.analysis_results[f"{file_id}_ml_{target_column}"] = results
            return results
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def perform_clustering(self, file_id: str, n_clusters: int = None) -> Dict:
        """Perform clustering analysis on the dataset"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered for data analysis"}
        
        try:
            df = self.data_files[file_id]['dataframe']
            ml_trainer = MLModelTrainer(df)
            
            clustering_results = ml_trainer.perform_clustering(n_clusters)
            
            results = {
                "status": "success",
                "file_id": file_id,
                "clustering_results": clustering_results,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.analysis_results[f"{file_id}_clustering"] = results
            return results
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def generate_gan_visualization(self, file_id: str, columns: List[str] = None) -> Dict:
        """Generate GAN-based data visualization"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered for data analysis"}
        
        try:
            df = self.data_files[file_id]['dataframe']
            gan_generator = GANDiagramGenerator()
            
            gan_results = gan_generator.generate_synthetic_data_visualization(df, columns)
            
            results = {
                "status": "success",
                "file_id": file_id,
                "gan_results": gan_results,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.analysis_results[f"{file_id}_gan"] = results
            return results
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def create_advanced_visualizations(self, file_id: str, chart_types: List[str] = None) -> Dict:
        """Create advanced data visualizations"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered for data analysis"}
        
        try:
            df = self.data_files[file_id]['dataframe']
            visualizations = {}
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not chart_types:
                chart_types = ["pairplot", "parallel_coordinates", "3d_scatter", "sunburst", "treemap"]
            
            # 1. Pairplot for numeric variables
            if "pairplot" in chart_types and len(numeric_cols) >= 2:
                # Create a correlation scatter matrix
                fig = make_subplots(
                    rows=len(numeric_cols), cols=len(numeric_cols),
                    subplot_titles=[f"{col1} vs {col2}" for col1 in numeric_cols for col2 in numeric_cols]
                )
                
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i == j:
                            # Diagonal: histogram
                            fig.add_trace(
                                go.Histogram(x=df[col1], name=col1, showlegend=False),
                                row=i+1, col=j+1
                            )
                        else:
                            # Off-diagonal: scatter plot
                            fig.add_trace(
                                go.Scatter(x=df[col2], y=df[col1], mode='markers', 
                                         name=f"{col1} vs {col2}", showlegend=False),
                                row=i+1, col=j+1
                            )
                
                fig.update_layout(title="Pairplot of Numeric Variables", height=200*len(numeric_cols))
                visualizations["pairplot"] = fig.to_json()
            
            # 2. Parallel coordinates plot
            if "parallel_coordinates" in chart_types and len(numeric_cols) >= 3:
                fig = go.Figure(data=
                    go.Parcoords(
                        line=dict(color=df[numeric_cols[0]], colorscale='Viridis'),
                        dimensions=[
                            dict(range=[df[col].min(), df[col].max()],
                                label=col, values=df[col])
                            for col in numeric_cols[:6]  # Limit to 6 dimensions
                        ]
                    )
                )
                fig.update_layout(title="Parallel Coordinates Plot")
                visualizations["parallel_coordinates"] = fig.to_json()
            
            # 3. 3D scatter plot
            if "3d_scatter" in chart_types and len(numeric_cols) >= 3:
                fig = px.scatter_3d(
                    df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2],
                    color=categorical_cols[0] if categorical_cols else None,
                    title="3D Scatter Plot"
                )
                visualizations["3d_scatter"] = fig.to_json()
            
            # 4. Sunburst chart for categorical data
            if "sunburst" in chart_types and len(categorical_cols) >= 2:
                # Create a sunburst chart with categorical variables
                df_sunburst = df[categorical_cols[:3]].value_counts().reset_index()
                df_sunburst.columns = list(categorical_cols[:3]) + ['count']
                
                fig = px.sunburst(
                    df_sunburst,
                    path=categorical_cols[:3],
                    values='count',
                    title="Sunburst Chart of Categorical Variables"
                )
                visualizations["sunburst"] = fig.to_json()
            
            # 5. Treemap
            if "treemap" in chart_types and categorical_cols:
                df_treemap = df[categorical_cols[0]].value_counts().reset_index()
                df_treemap.columns = [categorical_cols[0], 'count']
                
                fig = px.treemap(
                    df_treemap,
                    path=[categorical_cols[0]],
                    values='count',
                    title=f"Treemap of {categorical_cols[0]}"
                )
                visualizations["treemap"] = fig.to_json()
            
            results = {
                "status": "success",
                "file_id": file_id,
                "visualizations": visualizations,
                "chart_types_created": list(visualizations.keys()),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            self.analysis_results[f"{file_id}_advanced_viz"] = results
            return results
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_data_insights(self, file_id: str) -> Dict:
        """Generate AI-powered insights about the dataset"""
        if file_id not in self.data_files:
            return {"status": "error", "message": "File not registered for data analysis"}
        
        try:
            df = self.data_files[file_id]['dataframe']
            
            insights = {
                "dataset_overview": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                    "duplicate_rows": df.duplicated().sum(),
                    "missing_values_total": df.isnull().sum().sum()
                },
                "column_insights": {},
                "data_quality": {},
                "recommendations": []
            }
            
            # Column-wise insights
            for col in df.columns:
                col_info = {
                    "data_type": str(df[col].dtype),
                    "unique_values": df[col].nunique(),
                    "missing_percentage": round(df[col].isnull().sum() / len(df) * 100, 2)
                }
                
                if df[col].dtype in ['int64', 'float64']:
                    col_info.update({
                        "mean": round(df[col].mean(), 3),
                        "median": round(df[col].median(), 3),
                        "std": round(df[col].std(), 3),
                        "skewness": round(df[col].skew(), 3),
                        "kurtosis": round(df[col].kurtosis(), 3)
                    })
                else:
                    mode_value = df[col].mode()
                    col_info.update({
                        "most_frequent": mode_value.iloc[0] if not mode_value.empty else "N/A",
                        "frequency_of_mode": df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
                    })
                
                insights["column_insights"][col] = col_info
            
            # Data quality assessment
            insights["data_quality"] = {
                "completeness_score": round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
                "uniqueness_score": round((len(df) - df.duplicated().sum()) / len(df) * 100, 2),
                "columns_with_high_missing": [
                    col for col in df.columns 
                    if df[col].isnull().sum() / len(df) > 0.1
                ]
            }
            
            # Generate recommendations
            recommendations = []
            
            if df.duplicated().sum() > 0:
                recommendations.append(f"Consider removing {df.duplicated().sum()} duplicate rows")
            
            high_missing_cols = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.2]
            if high_missing_cols:
                recommendations.append(f"Columns with high missing values (>20%): {', '.join(high_missing_cols)}")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                recommendations.append("Dataset suitable for regression/classification tasks")
                recommendations.append("Consider feature scaling for machine learning models")
            
            if len(df.select_dtypes(include=['object']).columns) > 0:
                recommendations.append("Consider encoding categorical variables for ML models")
            
            insights["recommendations"] = recommendations
            
            return {
                "status": "success",
                "file_id": file_id,
                "insights": insights,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_analysis_history(self, file_id: str = None) -> Dict:
        """Get history of all analyses performed"""
        if file_id:
            file_analyses = {
                key: value for key, value in self.analysis_results.items() 
                if key.startswith(file_id)
            }
            return {"file_id": file_id, "analyses": file_analyses}
        
        return {"all_analyses": self.analysis_results}
    
    def export_results(self, analysis_key: str, format_type: str = "json") -> Dict:
        """Export analysis results in specified format"""
        if analysis_key not in self.analysis_results:
            return {"status": "error", "message": "Analysis not found"}
        
        results = self.analysis_results[analysis_key]
        
        if format_type == "json":
            return {
                "status": "success",
                "data": results,
                "export_format": "json"
            }
        elif format_type == "csv":
            # For certain results, convert to CSV format
            if "summary_statistics" in results:
                # Extract summary stats and convert to CSV-like structure
                return {
                    "status": "success", 
                    "message": "Summary statistics extracted",
                    "export_format": "csv"
                }
        
        return {"status": "error", "message": f"Unsupported export format: {format_type}"}
    
    def cleanup_analysis(self, file_id: str) -> Dict:
        """Clean up analysis data for a specific file"""
        try:
            # Remove from data files
            if file_id in self.data_files:
                del self.data_files[file_id]
            
            # Remove related analyses
            keys_to_remove = [key for key in self.analysis_results.keys() if key.startswith(file_id)]
            for key in keys_to_remove:
                del self.analysis_results[key]
            
            return {
                "status": "success",
                "message": f"Cleaned up analysis data for file {file_id}",
                "removed_analyses": len(keys_to_remove)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}