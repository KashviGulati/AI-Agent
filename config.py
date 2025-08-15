"""
config.py - Enhanced Configuration for RAG System with Data Analysis
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR = BASE_DIR / "uploaded_files"
DATA_ANALYSIS_DIR = BASE_DIR / "data_analysis_results"
METADATA_FILE = BASE_DIR / "file_metadata.json"

# Ensure directories exist
for directory in [CHROMA_DIR, UPLOAD_DIR, DATA_ANALYSIS_DIR]:
    directory.mkdir(exist_ok=True)

# RAG System Configuration
RAG_CONFIG = {
    "collection_name": "document_store",
    "embed_model": "all-MiniLM-L6-v2",
    "gemini_model": "gemini-1.5-flash",
    "chunk_size": 800,
    "chunk_overlap": 200,
    "top_k": 10,
    "relevance_threshold": 1.2,
}

# Data Analysis Configuration
DATA_ANALYSIS_CONFIG = {
    "supported_formats": [".csv", ".xlsx", ".xls"],
    "max_file_size_mb": 100,
    "default_test_size": 0.2,
    "random_state": 42,
    "max_clusters": 10,
    "min_samples_for_clustering": 50,
    "correlation_threshold": 0.7,
    "outlier_threshold": 1.5,  # IQR multiplier
    "missing_value_threshold": 0.1,  # 10% threshold for high missing values
}

# Machine Learning Configuration
ML_CONFIG = {
    "classification_models": {
        "Random Forest": {
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": None
        },
        "Logistic Regression": {
            "random_state": 42,
            "max_iter": 1000,
            "solver": "lbfgs"
        }
    },
    "regression_models": {
        "Random Forest": {
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": None
        },
        "Linear Regression": {}
    },
    "clustering_algorithms": {
        "KMeans": {
            "random_state": 42,
            "n_init": 10,
            "max_iter": 300
        }
    },
    "preprocessing": {
        "scale_features": True,
        "handle_categorical": True,
        "impute_missing": True
    }
}

# GAN Configuration
GAN_CONFIG = {
    "noise_dimension": 10,
    "hidden_dimensions": [128, 256],
    "learning_rate": 0.0002,
    "batch_size": 32,
    "epochs": 100,
    "beta1": 0.5,
    "beta2": 0.999,
    "device": "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "default_color_palette": "husl",
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "max_categories_in_plot": 10,
    "max_numeric_columns_in_pairplot": 6,
    "plotly_theme": "plotly_white",
    "chart_types": {
        "correlation_heatmap": True,
        "distribution_plots": True,
        "box_plots": True,
        "pairplot": True,
        "parallel_coordinates": True,
        "3d_scatter": True,
        "sunburst": True,
        "treemap": True
    }
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "workers": 1,
    "log_level": "info",
    "cors_origins": ["*"],
    "max_upload_size": 100 * 1024 * 1024,  # 100MB
    "timeout": 300,  # 5 minutes
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
    "log_file": BASE_DIR / "system.log",
    "max_log_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_size": 1000,
    "parallel_processing": True,
    "max_workers": 4,
    "memory_limit_gb": 8,
    "gpu_memory_fraction": 0.8 if os.environ.get("CUDA_AVAILABLE") else 0
}

# Security Configuration
SECURITY_CONFIG = {
    "allowed_file_types": {
        "documents": [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt"],
        "data": [".csv", ".xlsx", ".xls"]
    },
    "max_file_size": {
        "documents": 50 * 1024 * 1024,  # 50MB
        "data": 100 * 1024 * 1024       # 100MB
    },
    "scan_uploads": True,
    "quarantine_suspicious": True
}

# Feature Flags
FEATURE_FLAGS = {
    "enable_rag": True,
    "enable_data_analysis": True,
    "enable_ml_training": True,
    "enable_clustering": True,
    "enable_gan_visualization": True,
    "enable_advanced_visualizations": True,
    "enable_batch_processing": True,
    "enable_model_caching": True,
    "enable_result_export": True,
    "enable_analysis_history": True
}

# Environment-specific overrides
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    API_CONFIG["reload"] = False
    API_CONFIG["workers"] = 4
    LOGGING_CONFIG["level"] = "WARNING"
    PERFORMANCE_CONFIG["max_workers"] = 8
    
elif ENVIRONMENT == "development":
    API_CONFIG["reload"] = True
    API_CONFIG["workers"] = 1
    LOGGING_CONFIG["level"] = "DEBUG"
    PERFORMANCE_CONFIG["max_workers"] = 2

elif ENVIRONMENT == "testing":
    API_CONFIG["port"] = 8001
    DATA_ANALYSIS_CONFIG["max_file_size_mb"] = 10
    ML_CONFIG["classification_models"]["Random Forest"]["n_estimators"] = 10
    GAN_CONFIG["epochs"] = 10

def get_config(section: str = None):
    """Get configuration for a specific section or all configurations"""
    configs = {
        "rag": RAG_CONFIG,
        "data_analysis": DATA_ANALYSIS_CONFIG,
        "ml": ML_CONFIG,
        "gan": GAN_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "api": API_CONFIG,
        "logging": LOGGING_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "security": SECURITY_CONFIG,
        "features": FEATURE_FLAGS
    }
    
    if section:
        return configs.get(section, {})
    return configs

def update_config(section: str, updates: dict):
    """Update configuration for a specific section"""
    configs = get_config()
    if section in configs:
        configs[section].update(updates)
        # In a real implementation, you might want to save this to a file
        return True
    return False

# Export commonly used configurations
__all__ = [
    "RAG_CONFIG",
    "DATA_ANALYSIS_CONFIG", 
    "ML_CONFIG",
    "GAN_CONFIG",
    "VISUALIZATION_CONFIG",
    "API_CONFIG",
    "LOGGING_CONFIG",
    "PERFORMANCE_CONFIG",
    "SECURITY_CONFIG",
    "FEATURE_FLAGS",
    "get_config",
    "update_config",
    "BASE_DIR",
    "CHROMA_DIR",
    "UPLOAD_DIR",
    "DATA_ANALYSIS_DIR",
    "METADATA_FILE"
]