"""
Machine Learning Model Comparison for Cognitive Fatigue Detection
Implements multiple models for ISEF-level comparative analysis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import json
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class FatigueClassifier:
    """
    Multi-model classifier for cognitive fatigue detection.
    Compares traditional ML and neural network approaches.
    """
    
    def __init__(self):
        """Initialize all models for comparison."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                max_iter=500,
                random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.results = {}
        
    def prepare_data(self, features_df: pd.DataFrame, labels: np.ndarray,
                    test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            features_df: DataFrame with extracted features
            labels: Binary labels (0=focused, 1=fatigued)
            test_size: Proportion for test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Remove non-feature columns
        feature_cols = [col for col in features_df.columns 
                       if col not in ['window_start', 'window_end']]
        X = features_df[feature_cols].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train all models and return training metrics.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of training results
        """
        training_results = {}
        
        print("Training Models...")
        print("=" * 60)
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='accuracy')
            
            training_results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"  Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return training_results
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comprehensive metrics for each model
        """
        results = []
        
        print("\n" + "=" * 60)
        print("Model Evaluation Results")
        print("=" * 60)
        
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            }
            
            if y_prob is not None:
                metrics['ROC AUC'] = roc_auc_score(y_test, y_prob)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['True Negatives'] = cm[0, 0]
            metrics['False Positives'] = cm[0, 1]
            metrics['False Negatives'] = cm[1, 0]
            metrics['True Positives'] = cm[1, 1]
            
            # Specificity
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            metrics['Specificity'] = specificity
            
            results.append(metrics)
            
            # Print results
            print(f"\n{name}:")
            print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"  Precision: {metrics['Precision']:.4f}")
            print(f"  Recall:    {metrics['Recall']:.4f}")
            print(f"  F1 Score:  {metrics['F1 Score']:.4f}")
            if 'ROC AUC' in metrics:
                print(f"  ROC AUC:   {metrics['ROC AUC']:.4f}")
        
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        return results_df
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance rankings
        """
        importance_data = []
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                for feat, imp in zip(feature_names, importances):
                    importance_data.append({
                        'Model': name,
                        'Feature': feat,
                        'Importance': imp
                    })
        
        if importance_data:
            return pd.DataFrame(importance_data)
        else:
            return pd.DataFrame()
    
    def save_best_model(self, filepath: str = '/home/claude/best_model.pkl'):
        """
        Save the best performing model.
        
        Args:
            filepath: Path to save model
        """
        if self.results.empty:
            print("No models have been evaluated yet.")
            return
        
        # Find best model by accuracy
        best_idx = self.results['Accuracy'].idxmax()
        best_model_name = self.results.loc[best_idx, 'Model']
        best_model = self.models[best_model_name]
        
        # Save model and scaler
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'model_name': best_model_name,
            'accuracy': self.results.loc[best_idx, 'Accuracy']
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nBest model ({best_model_name}) saved to {filepath}")
        print(f"Accuracy: {model_data['accuracy']:.4f}")
        
        return filepath
    
    def load_model(self, filepath: str):
        """Load a saved model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        return model_data['model'], model_data['model_name']
    
    def predict_realtime(self, features: np.ndarray, model_name: str = None) -> Tuple[int, float]:
        """
        Make real-time prediction on new features.
        
        Args:
            features: Feature vector
            model_name: Specific model to use (None = best model)
            
        Returns:
            (predicted_class, confidence)
        """
        if model_name is None:
            # Use best model
            if not self.results.empty:
                best_idx = self.results['Accuracy'].idxmax()
                model_name = self.results.loc[best_idx, 'Model']
            else:
                model_name = list(self.models.keys())[0]
        
        model = self.models[model_name]
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        if hasattr(model, 'predict_proba'):
            confidence = model.predict_proba(features_scaled)[0][prediction]
        else:
            confidence = 1.0  # SVM without probability
        
        return prediction, confidence


def create_synthetic_dataset(n_samples: int = 200) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create synthetic EEG feature dataset for testing.
    
    Args:
        n_samples: Number of samples per class
        
    Returns:
        (features_df, labels)
    """
    from eeg_processor import EEGProcessor, simulate_eeg_data
    
    processor = EEGProcessor(sampling_rate=256)
    
    all_features = []
    all_labels = []
    
    print(f"Generating synthetic dataset ({n_samples * 2} samples)...")
    
    # Generate focused samples
    for i in range(n_samples):
        eeg_data = simulate_eeg_data(duration=4, state='focused')
        features = processor.extract_features(eeg_data)
        all_features.append(features)
        all_labels.append(0)  # 0 = focused
    
    # Generate fatigued samples
    for i in range(n_samples):
        eeg_data = simulate_eeg_data(duration=4, state='fatigued')
        features = processor.extract_features(eeg_data)
        all_features.append(features)
        all_labels.append(1)  # 1 = fatigued
    
    features_df = pd.DataFrame(all_features)
    labels = np.array(all_labels)
    
    print(f"Dataset created: {len(features_df)} samples, {len(features_df.columns)} features")
    
    return features_df, labels


if __name__ == "__main__":
    print("Cognitive Fatigue Detection - Model Comparison")
    print("=" * 60)
    
    # Create synthetic dataset
    features_df, labels = create_synthetic_dataset(n_samples=200)
    
    # Initialize classifier
    classifier = FatigueClassifier()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(features_df, labels)
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Focused: {np.sum(labels == 0)}, Fatigued: {np.sum(labels == 1)}")
    
    # Train all models
    training_results = classifier.train_all_models(X_train, y_train)
    
    # Evaluate all models
    results_df = classifier.evaluate_all_models(X_test, y_test)
    
    # Save results
    results_df.to_csv('/home/claude/model_comparison_results.csv', index=False)
    print("\nResults saved to model_comparison_results.csv")
    
    # Save best model
    classifier.save_best_model()
