"""
EEG Signal Processing and Feature Extraction Module
For Cognitive Fatigue Detection Project
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
import pandas as pd
from typing import Tuple, Dict, List


class EEGProcessor:
    """
    Processes raw EEG signals and extracts cognitive fatigue features.
    
    Key Features:
    - Band power extraction (alpha, beta, theta, delta)
    - Engagement ratios (beta/alpha, beta/theta)
    - Statistical features for ML classification
    """
    
    def __init__(self, sampling_rate: int = 256):
        """
        Initialize EEG processor.
        
        Args:
            sampling_rate: Sampling frequency in Hz (256 for most devices)
        """
        self.fs = sampling_rate
        
        # Define EEG frequency bands (Hz)
        self.bands = {
            'delta': (0.5, 4),    # Deep sleep
            'theta': (4, 8),       # Drowsiness, fatigue
            'alpha': (8, 13),      # Relaxed, closed eyes
            'beta': (13, 30),      # Active thinking, focus
            'gamma': (30, 50)      # High-level cognition
        }
        
    def bandpass_filter(self, data: np.ndarray, low: float, high: float, 
                       order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to isolate frequency band.
        
        Args:
            data: Raw EEG signal
            low: Lower frequency bound (Hz)
            high: Upper frequency bound (Hz)
            order: Filter order
            
        Returns:
            Filtered signal
        """
        nyquist = self.fs / 2
        low_norm = low / nyquist
        high_norm = high / nyquist
        
        # Butterworth bandpass filter
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        filtered = signal.filtfilt(b, a, data)
        
        return filtered
    
    def compute_band_power(self, data: np.ndarray, band: Tuple[float, float]) -> float:
        """
        Compute average power in a frequency band using Welch's method.
        
        Args:
            data: EEG signal segment
            band: Frequency range (low, high) in Hz
            
        Returns:
            Average band power
        """
        # Welch's method for power spectral density
        freqs, psd = signal.welch(data, self.fs, nperseg=min(256, len(data)))
        
        # Find frequencies within band
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        band_power = np.trapz(psd[band_mask], freqs[band_mask])
        
        return band_power
    
    def extract_features(self, eeg_segment: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive feature set for cognitive fatigue classification.
        
        KEY FEATURES FOR FATIGUE DETECTION:
        - Beta/Alpha ratio: Decreases with fatigue
        - Beta/Theta ratio: Decreases with fatigue
        - Theta power: Increases with fatigue
        - Alpha power: May increase with drowsiness
        
        Args:
            eeg_segment: Single-channel EEG data (typically 2-10 seconds)
            
        Returns:
            Dictionary of features for ML model
        """
        features = {}
        
        # 1. Band powers (absolute)
        for band_name, band_range in self.bands.items():
            power = self.compute_band_power(eeg_segment, band_range)
            features[f'{band_name}_power'] = power
        
        # 2. Relative band powers (normalized)
        total_power = sum(features[f'{b}_power'] for b in self.bands.keys())
        for band_name in self.bands.keys():
            features[f'{band_name}_relative'] = features[f'{band_name}_power'] / (total_power + 1e-10)
        
        # 3. CRITICAL RATIOS for fatigue detection
        features['beta_alpha_ratio'] = features['beta_power'] / (features['alpha_power'] + 1e-10)
        features['beta_theta_ratio'] = features['beta_power'] / (features['theta_power'] + 1e-10)
        features['engagement_index'] = features['beta_power'] / (features['alpha_power'] + features['theta_power'] + 1e-10)
        features['theta_beta_ratio'] = features['theta_power'] / (features['beta_power'] + 1e-10)
        
        # 4. Statistical features (time domain)
        features['mean_amplitude'] = np.mean(eeg_segment)
        features['std_amplitude'] = np.std(eeg_segment)
        features['skewness'] = skew(eeg_segment)
        features['kurtosis_val'] = kurtosis(eeg_segment)
        features['peak_to_peak'] = np.ptp(eeg_segment)
        
        # 5. Spectral entropy (complexity measure)
        freqs, psd = signal.welch(eeg_segment, self.fs, nperseg=min(256, len(eeg_segment)))
        psd_norm = psd / (np.sum(psd) + 1e-10)
        features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        return features
    
    def process_continuous_data(self, eeg_data: np.ndarray, 
                               window_size: float = 4.0,
                               overlap: float = 2.0) -> pd.DataFrame:
        """
        Process continuous EEG data into feature windows.
        
        Args:
            eeg_data: Continuous EEG signal
            window_size: Window length in seconds
            overlap: Overlap between windows in seconds
            
        Returns:
            DataFrame with features for each window
        """
        window_samples = int(window_size * self.fs)
        step_samples = int((window_size - overlap) * self.fs)
        
        feature_list = []
        
        # Slide window across data
        for start in range(0, len(eeg_data) - window_samples, step_samples):
            end = start + window_samples
            segment = eeg_data[start:end]
            
            # Extract features
            features = self.extract_features(segment)
            features['window_start'] = start / self.fs
            features['window_end'] = end / self.fs
            
            feature_list.append(features)
        
        return pd.DataFrame(feature_list)
    
    def denoise_signal(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Remove common noise artifacts from EEG signal.
        
        - 50/60 Hz powerline interference
        - Low-frequency drift
        - High-frequency muscle noise
        
        Args:
            eeg_data: Raw EEG signal
            
        Returns:
            Cleaned signal
        """
        # Remove powerline noise (50 Hz or 60 Hz)
        b_notch, a_notch = signal.iirnotch(60.0, 30.0, self.fs)
        cleaned = signal.filtfilt(b_notch, a_notch, eeg_data)
        
        # High-pass filter to remove drift (>0.5 Hz)
        b_high, a_high = signal.butter(4, 0.5 / (self.fs / 2), btype='high')
        cleaned = signal.filtfilt(b_high, a_high, cleaned)
        
        # Low-pass filter to remove muscle noise (<50 Hz)
        b_low, a_low = signal.butter(4, 50.0 / (self.fs / 2), btype='low')
        cleaned = signal.filtfilt(b_low, a_low, cleaned)
        
        return cleaned


def simulate_eeg_data(duration: int = 60, sampling_rate: int = 256, 
                     state: str = 'focused') -> np.ndarray:
    """
    Generate synthetic EEG data for testing (mimics real patterns).
    
    This is for demonstration purposes. For ISEF, you'll use real datasets.
    
    Args:
        duration: Length in seconds
        sampling_rate: Sampling frequency (Hz)
        state: 'focused' or 'fatigued'
        
    Returns:
        Simulated EEG signal
    """
    n_samples = duration * sampling_rate
    t = np.linspace(0, duration, n_samples)
    
    if state == 'focused':
        # Higher beta (15-20 Hz), lower theta (5-7 Hz)
        beta = 8 * np.sin(2 * np.pi * 17 * t + np.random.randn() * 0.5)
        alpha = 5 * np.sin(2 * np.pi * 10 * t + np.random.randn() * 0.5)
        theta = 2 * np.sin(2 * np.pi * 6 * t + np.random.randn() * 0.5)
    else:  # fatigued
        # Lower beta, higher theta
        beta = 3 * np.sin(2 * np.pi * 17 * t + np.random.randn() * 0.5)
        alpha = 7 * np.sin(2 * np.pi * 10 * t + np.random.randn() * 0.5)
        theta = 6 * np.sin(2 * np.pi * 6 * t + np.random.randn() * 0.5)
    
    # Combine signals + add noise
    eeg_signal = beta + alpha + theta + np.random.randn(n_samples) * 0.5
    
    return eeg_signal


if __name__ == "__main__":
    # Test the processor
    print("Testing EEG Processor...")
    
    processor = EEGProcessor(sampling_rate=256)
    
    # Generate test data
    focused_data = simulate_eeg_data(duration=10, state='focused')
    fatigued_data = simulate_eeg_data(duration=10, state='fatigued')
    
    # Process
    focused_features_df = processor.process_continuous_data(focused_data)
    fatigued_features_df = processor.process_continuous_data(fatigued_data)
    
    print(f"\nExtracted {len(focused_features_df)} windows from focused state")
    print(f"Features per window: {len(focused_features_df.columns)}")
    print("\nKey features (focused state):")
    print(focused_features_df[['beta_alpha_ratio', 'beta_theta_ratio', 'theta_power']].describe())
    
    print("\nKey features (fatigued state):")
    print(fatigued_features_df[['beta_alpha_ratio', 'beta_theta_ratio', 'theta_power']].describe())
