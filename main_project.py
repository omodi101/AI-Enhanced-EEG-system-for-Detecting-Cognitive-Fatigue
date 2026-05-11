"""
MASTER CONTROLLER - Cognitive Fatigue Detection System
Orchestrates the complete ISEF project pipeline
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Import all project modules
from eeg_processor import EEGProcessor, simulate_eeg_data
from ml_models import FatigueClassifier, create_synthetic_dataset
from edge_computing import EdgeFatigueDetector, benchmark_edge_performance
from visualizations import FatigueVisualizer


class ISEFProject:
    """
    Complete ISEF project orchestrator.
    Runs full experimental pipeline from data generation to results.
    """
    
    def __init__(self, output_dir: str = '/home/claude'):
        """
        Initialize ISEF project.
        
        Args:
            output_dir: Directory for all outputs
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.processor = EEGProcessor(sampling_rate=256)
        self.classifier = FatigueClassifier()
        self.visualizer = FatigueVisualizer()
        
        # Results storage
        self.results = {}
        
        print("=" * 70)
        print("COGNITIVE FATIGUE DETECTION - ISEF PROJECT")
        print("AI-Enhanced Edge Computing System")
        print("=" * 70)
        print(f"\nInitialized: {datetime.now()}")
        print(f"Output directory: {output_dir}")
    
    def step_1_generate_dataset(self, n_samples: int = 200):
        """
        Step 1: Generate or load EEG dataset.
        
        Args:
            n_samples: Number of samples per class
        """
        print("\n" + "=" * 70)
        print("STEP 1: DATASET GENERATION")
        print("=" * 70)
        
        # Create synthetic dataset (in real project, use actual EEG data)
        features_df, labels = create_synthetic_dataset(n_samples=n_samples)
        
        self.features_df = features_df
        self.labels = labels
        
        # Save dataset
        dataset_path = os.path.join(self.output_dir, 'dataset_features.csv')
        features_df['label'] = labels
        features_df.to_csv(dataset_path, index=False)
        
        print(f"\n✓ Dataset saved: {dataset_path}")
        print(f"  Total samples: {len(features_df)}")
        print(f"  Features: {len(features_df.columns) - 1}")
        print(f"  Focused: {np.sum(labels == 0)}, Fatigued: {np.sum(labels == 1)}")
        
        # Generate sample EEG signals for visualization
        self.focused_signal = simulate_eeg_data(duration=10, state='focused')
        self.fatigued_signal = simulate_eeg_data(duration=10, state='fatigued')
        
        self.results['dataset'] = {
            'n_samples': len(features_df),
            'n_features': len(features_df.columns) - 1,
            'n_focused': int(np.sum(labels == 0)),
            'n_fatigued': int(np.sum(labels == 1))
        }
        
        return features_df, labels
    
    def step_2_train_models(self):
        """
        Step 2: Train and compare multiple ML models.
        """
        print("\n" + "=" * 70)
        print("STEP 2: MODEL TRAINING & COMPARISON")
        print("=" * 70)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.classifier.prepare_data(
            self.features_df, self.labels, test_size=0.2
        )
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")
        
        # Train all models
        training_results = self.classifier.train_all_models(X_train, y_train)
        
        # Evaluate on test set
        self.results_df = self.classifier.evaluate_all_models(X_test, y_test)
        
        # Get feature importance
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['window_start', 'window_end', 'label']]
        self.importance_df = self.classifier.get_feature_importance(feature_cols)
        
        # Save best model
        model_path = self.classifier.save_best_model(
            filepath=os.path.join(self.output_dir, 'best_model.pkl')
        )
        
        # Save results
        results_path = os.path.join(self.output_dir, 'model_comparison_results.csv')
        self.results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved: {results_path}")
        
        if not self.importance_df.empty:
            importance_path = os.path.join(self.output_dir, 'feature_importance.csv')
            self.importance_df.to_csv(importance_path, index=False)
            print(f"✓ Feature importance saved: {importance_path}")
        
        self.results['models'] = {
            'best_model': self.results_df.loc[self.results_df['Accuracy'].idxmax(), 'Model'],
            'best_accuracy': float(self.results_df['Accuracy'].max()),
            'mean_accuracy': float(self.results_df['Accuracy'].mean())
        }
        
        return self.results_df
    
    def step_3_edge_computing(self, n_iterations: int = 100):
        """
        Step 3: Test edge computing performance.
        
        Args:
            n_iterations: Number of benchmark iterations
        """
        print("\n" + "=" * 70)
        print("STEP 3: EDGE COMPUTING PERFORMANCE")
        print("=" * 70)
        
        model_path = os.path.join(self.output_dir, 'best_model.pkl')
        
        # Benchmark performance
        self.latencies = benchmark_edge_performance(
            model_path, 
            n_iterations=n_iterations
        )
        
        # Save latency data
        latency_path = os.path.join(self.output_dir, 'edge_latencies.csv')
        pd.DataFrame({'latency_ms': self.latencies}).to_csv(latency_path, index=False)
        print(f"\n✓ Latency data saved: {latency_path}")
        
        self.results['edge_computing'] = {
            'mean_latency_ms': float(np.mean(self.latencies)),
            'median_latency_ms': float(np.median(self.latencies)),
            'p95_latency_ms': float(np.percentile(self.latencies, 95)),
            'p99_latency_ms': float(np.percentile(self.latencies, 99)),
            'max_latency_ms': float(np.max(self.latencies)),
            'real_time_capable': bool(np.mean(self.latencies) < 100),
            'throughput_per_sec': float(1000 / np.mean(self.latencies))
        }
        
        return self.latencies
    
    def step_4_visualizations(self):
        """
        Step 4: Generate all ISEF-ready visualizations.
        """
        print("\n" + "=" * 70)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        # 1. Model comparison
        print("\n1. Creating model comparison plot...")
        self.visualizer.plot_model_comparison(
            self.results_df,
            save_path=os.path.join(self.output_dir, 'model_comparison.png')
        )
        
        # 2. Feature importance (if available)
        if not self.importance_df.empty:
            print("2. Creating feature importance plot...")
            self.visualizer.plot_feature_importance(
                self.importance_df,
                save_path=os.path.join(self.output_dir, 'feature_importance.png')
            )
        
        # 3. EEG signals
        print("3. Creating EEG signal analysis plot...")
        self.visualizer.plot_eeg_signals(
            self.focused_signal,
            self.fatigued_signal,
            save_path=os.path.join(self.output_dir, 'eeg_signals.png')
        )
        
        # 4. Edge performance
        print("4. Creating edge performance plot...")
        self.visualizer.plot_edge_performance(
            self.latencies,
            save_path=os.path.join(self.output_dir, 'edge_performance.png')
        )
        
        print("\n✓ All visualizations generated successfully!")
    
    def step_5_generate_report(self):
        """
        Step 5: Generate comprehensive ISEF report.
        """
        print("\n" + "=" * 70)
        print("STEP 5: GENERATING ISEF REPORT")
        print("=" * 70)
        
        # Generate text report
        report = self.visualizer.generate_isef_report(
            self.results_df,
            self.latencies,
            save_path=os.path.join(self.output_dir, 'isef_report.txt')
        )
        
        # Save JSON results
        import json
        json_path = os.path.join(self.output_dir, 'results_summary.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ JSON results saved: {json_path}")
        
        return report
    
    def run_complete_pipeline(self, n_samples: int = 200, 
                             n_edge_iterations: int = 100):
        """
        Run the complete ISEF project pipeline.
        
        Args:
            n_samples: Number of samples per class for dataset
            n_edge_iterations: Number of edge benchmark iterations
        """
        print("\n" + "=" * 70)
        print("RUNNING COMPLETE ISEF PROJECT PIPELINE")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Dataset size: {n_samples * 2} samples")
        print(f"  Edge iterations: {n_edge_iterations}")
        print(f"  Output: {self.output_dir}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Generate dataset
            self.step_1_generate_dataset(n_samples=n_samples)
            
            # Step 2: Train models
            self.step_2_train_models()
            
            # Step 3: Edge computing
            self.step_3_edge_computing(n_iterations=n_edge_iterations)
            
            # Step 4: Visualizations
            self.step_4_visualizations()
            
            # Step 5: Generate report
            self.step_5_generate_report()
            
            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 70)
            print("PIPELINE COMPLETE!")
            print("=" * 70)
            print(f"\nExecution time: {duration:.2f} seconds")
            print(f"\nKey Results:")
            print(f"  Best Model: {self.results['models']['best_model']}")
            print(f"  Accuracy: {self.results['models']['best_accuracy']:.4f}")
            print(f"  Mean Latency: {self.results['edge_computing']['mean_latency_ms']:.2f}ms")
            print(f"  Real-time Capable: {self.results['edge_computing']['real_time_capable']}")
            
            print(f"\nAll outputs saved to: {self.output_dir}")
            print("\nGenerated files:")
            print("  ├── dataset_features.csv")
            print("  ├── model_comparison_results.csv")
            print("  ├── feature_importance.csv")
            print("  ├── edge_latencies.csv")
            print("  ├── best_model.pkl")
            print("  ├── model_comparison.png")
            print("  ├── feature_importance.png")
            print("  ├── eeg_signals.png")
            print("  ├── edge_performance.png")
            print("  ├── isef_report.txt")
            print("  └── results_summary.json")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    Main entry point for ISEF project.
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  COGNITIVE FATIGUE DETECTION SYSTEM".center(68) + "║")
    print("║" + "  AI-Enhanced Edge Computing for EEG Analysis".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  ISEF-Level Science Fair Project".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")
    
    # Create project instance
    project = ISEFProject(output_dir='/home/claude')
    
    # Run complete pipeline
    success = project.run_complete_pipeline(
        n_samples=200,           # 400 total samples
        n_edge_iterations=100    # 100 latency measurements
    )
    
    if success:
        print("\n" + "=" * 70)
        print("SUCCESS! Your ISEF project is ready.")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Review the isef_report.txt for your findings")
        print("  2. Examine all visualization plots")
        print("  3. Use results_summary.json for quick reference")
        print("  4. Consider testing with real EEG data")
        print("\n")
        return 0
    else:
        print("\nPipeline failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
