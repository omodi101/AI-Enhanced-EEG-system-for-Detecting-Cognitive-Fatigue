"""
Edge Computing Module for Real-Time Cognitive Fatigue Detection
Optimized for deployment on Raspberry Pi, ESP32, or similar edge devices
"""

import numpy as np
import time
import pickle
from typing import Dict, Tuple
from collections import deque
import json


class EdgeFatigueDetector:
    """
    Lightweight real-time fatigue detector optimized for edge devices.
    
    Key optimizations:
    - Minimal memory footprint
    - Fast inference (<100ms)
    - Rolling window processing
    - No cloud dependency
    """
    
    def __init__(self, model_path: str, buffer_size: int = 1024):
        """
        Initialize edge detector.
        
        Args:
            model_path: Path to saved model
            buffer_size: Size of rolling buffer (samples)
        """
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_name = model_data['model_name']
        
        # Rolling buffer for continuous data
        self.buffer = deque(maxlen=buffer_size)
        self.sampling_rate = 256
        
        # Performance tracking
        self.inference_times = []
        self.prediction_history = deque(maxlen=100)
        
        print(f"Edge Detector Initialized: {self.model_name}")
        print(f"Buffer size: {buffer_size} samples ({buffer_size/self.sampling_rate:.2f} seconds)")
    
    def add_sample(self, sample: float):
        """
        Add single EEG sample to rolling buffer.
        
        Args:
            sample: Single EEG data point
        """
        self.buffer.append(sample)
    
    def add_batch(self, samples: np.ndarray):
        """
        Add multiple samples at once.
        
        Args:
            samples: Array of EEG samples
        """
        for s in samples:
            self.buffer.append(s)
    
    def extract_features_fast(self, data: np.ndarray) -> np.ndarray:
        """
        Fast feature extraction optimized for edge devices.
        Uses simplified computation to minimize latency.
        
        Args:
            data: EEG segment
            
        Returns:
            Feature vector
        """
        from scipy import signal
        
        features = []
        
        # Band power calculation (using FFT for speed)
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1/self.sampling_rate)
        power = np.abs(fft) ** 2
        
        # Define bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
        }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            band_powers[band_name] = np.sum(power[idx])
            features.append(band_powers[band_name])
        
        # Relative powers
        total_power = sum(band_powers.values()) + 1e-10
        for band_name in bands.keys():
            features.append(band_powers[band_name] / total_power)
        
        # Critical ratios
        features.append(band_powers['beta'] / (band_powers['alpha'] + 1e-10))  # beta/alpha
        features.append(band_powers['beta'] / (band_powers['theta'] + 1e-10))  # beta/theta
        features.append(band_powers['beta'] / (band_powers['alpha'] + band_powers['theta'] + 1e-10))  # engagement
        features.append(band_powers['theta'] / (band_powers['beta'] + 1e-10))  # theta/beta
        
        # Statistical features (fast)
        features.append(np.mean(data))
        features.append(np.std(data))
        features.append(np.ptp(data))  # peak-to-peak
        
        # Simplified spectral entropy
        psd_norm = power / (np.sum(power) + 1e-10)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        features.append(entropy)
        
        return np.array(features)
    
    def predict(self, return_features: bool = False) -> Tuple[int, float, float]:
        """
        Make real-time prediction from current buffer.
        
        Returns:
            (prediction, confidence, inference_time_ms)
        """
        if len(self.buffer) < self.sampling_rate:
            return -1, 0.0, 0.0  # Not enough data
        
        start_time = time.time()
        
        # Get data from buffer
        data = np.array(list(self.buffer))
        
        # Use most recent 4 seconds
        segment_length = 4 * self.sampling_rate
        if len(data) >= segment_length:
            data = data[-segment_length:]
        
        # Extract features
        features = self.extract_features_fast(data)
        
        # Scale
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        if hasattr(self.model, 'predict_proba'):
            confidence = self.model.predict_proba(features_scaled)[0][prediction]
        else:
            confidence = 1.0
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # Store prediction
        self.prediction_history.append({
            'timestamp': time.time(),
            'prediction': int(prediction),
            'confidence': float(confidence),
            'inference_time_ms': inference_time
        })
        
        if return_features:
            return prediction, confidence, inference_time, features
        
        return prediction, confidence, inference_time
    
    def get_performance_stats(self) -> Dict:
        """
        Get edge computing performance statistics.
        
        Returns:
            Dictionary with latency and throughput metrics
        """
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        
        stats = {
            'mean_latency_ms': np.mean(times),
            'median_latency_ms': np.median(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'std_latency_ms': np.std(times),
            'p95_latency_ms': np.percentile(times, 95),
            'p99_latency_ms': np.percentile(times, 99),
            'total_predictions': len(self.inference_times),
            'avg_fps': 1000 / np.mean(times) if np.mean(times) > 0 else 0
        }
        
        return stats
    
    def get_cognitive_state(self) -> Dict:
        """
        Get current cognitive state with trend analysis.
        
        Returns:
            Current state, confidence, and trend
        """
        if len(self.prediction_history) < 5:
            return {'state': 'Unknown', 'confidence': 0.0, 'trend': 'insufficient_data'}
        
        recent = list(self.prediction_history)[-10:]
        
        # Current state
        current = recent[-1]
        state_label = 'Fatigued' if current['prediction'] == 1 else 'Focused'
        
        # Calculate trend
        predictions = [p['prediction'] for p in recent]
        if len(predictions) >= 5:
            trend_score = np.mean(predictions[-5:])
            
            if trend_score > 0.7:
                trend = 'increasing_fatigue'
            elif trend_score < 0.3:
                trend = 'maintaining_focus'
            else:
                trend = 'variable'
        else:
            trend = 'unknown'
        
        return {
            'state': state_label,
            'confidence': current['confidence'],
            'trend': trend,
            'recent_predictions': predictions,
            'timestamp': current['timestamp']
        }
    
    def continuous_monitor(self, duration: int = 60, callback=None):
        """
        Run continuous monitoring with simulated data.
        
        Args:
            duration: Monitoring duration (seconds)
            callback: Optional callback function(state_dict)
        """
        from eeg_processor import simulate_eeg_data
        
        print(f"\nStarting {duration}s continuous monitoring...")
        print("=" * 60)
        
        start_time = time.time()
        prediction_count = 0
        
        while (time.time() - start_time) < duration:
            # Simulate incoming data (in real system, this comes from EEG device)
            new_samples = simulate_eeg_data(duration=1, state='focused' if np.random.rand() > 0.3 else 'fatigued')
            self.add_batch(new_samples)
            
            # Make prediction every second
            if len(self.buffer) >= self.sampling_rate:
                prediction, confidence, latency = self.predict()
                
                if prediction != -1:
                    prediction_count += 1
                    state = 'Fatigued' if prediction == 1 else 'Focused'
                    
                    print(f"[{prediction_count:3d}] State: {state:8s} | "
                          f"Confidence: {confidence:.3f} | "
                          f"Latency: {latency:.2f}ms")
                    
                    if callback:
                        callback({
                            'prediction': prediction,
                            'confidence': confidence,
                            'latency': latency
                        })
            
            time.sleep(0.5)  # Control loop rate
        
        # Print summary
        print("\n" + "=" * 60)
        print("Monitoring Complete")
        print("=" * 60)
        stats = self.get_performance_stats()
        
        print(f"\nPerformance Statistics:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Mean latency: {stats['mean_latency_ms']:.2f}ms")
        print(f"  Median latency: {stats['median_latency_ms']:.2f}ms")
        print(f"  95th percentile: {stats['p95_latency_ms']:.2f}ms")
        print(f"  99th percentile: {stats['p99_latency_ms']:.2f}ms")
        print(f"  Max latency: {stats['max_latency_ms']:.2f}ms")
        print(f"  Average FPS: {stats['avg_fps']:.2f}")


def benchmark_edge_performance(model_path: str, n_iterations: int = 100):
    """
    Benchmark edge computing performance.
    
    Args:
        model_path: Path to model
        n_iterations: Number of test iterations
    """
    from eeg_processor import simulate_eeg_data
    
    print("Edge Computing Performance Benchmark")
    print("=" * 60)
    
    detector = EdgeFatigueDetector(model_path)
    
    # Warm-up
    print("\nWarming up...")
    for _ in range(10):
        data = simulate_eeg_data(duration=4, state='focused')
        detector.add_batch(data)
        detector.predict()
    
    # Benchmark
    print(f"\nRunning {n_iterations} iterations...")
    
    latencies = []
    
    for i in range(n_iterations):
        # Simulate new data
        data = simulate_eeg_data(duration=4, state='focused')
        detector.add_batch(data)
        
        # Predict
        _, _, latency = detector.predict()
        latencies.append(latency)
        
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{n_iterations}")
    
    # Results
    latencies = np.array(latencies)
    
    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    print(f"Iterations: {n_iterations}")
    print(f"Mean latency: {np.mean(latencies):.2f}ms")
    print(f"Median latency: {np.median(latencies):.2f}ms")
    print(f"Min latency: {np.min(latencies):.2f}ms")
    print(f"Max latency: {np.max(latencies):.2f}ms")
    print(f"Std dev: {np.std(latencies):.2f}ms")
    print(f"95th percentile: {np.percentile(latencies, 95):.2f}ms")
    print(f"99th percentile: {np.percentile(latencies, 99):.2f}ms")
    print(f"\nThroughput: {1000/np.mean(latencies):.2f} predictions/second")
    print(f"Real-time capable: {'YES' if np.mean(latencies) < 100 else 'NO'} (<100ms target)")
    
    return latencies


if __name__ == "__main__":
    # Note: Run ml_models.py first to generate the model file
    
    import os
    model_path = '/home/claude/best_model.pkl'
    
    if os.path.exists(model_path):
        print("Testing Edge Computing System\n")
        
        # Benchmark performance
        latencies = benchmark_edge_performance(model_path, n_iterations=100)
        
        # Test continuous monitoring
        detector = EdgeFatigueDetector(model_path)
        detector.continuous_monitor(duration=30)
        
    else:
        print("Model file not found. Please run ml_models.py first.")
