import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pickle
import scipy.signal
import scipy.interpolate
from scipy.stats import norm
from scipy.fft import fft, fftfreq
from scipy.stats import ks_2samp
import threading
import time
from datetime import datetime
import warnings
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
warnings.filterwarnings('ignore')

class CircularConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv = Conv1D(filters, kernel_size, padding='valid', activation=activation)

    def call(self, inputs):
        padding_size = self.kernel_size // 2
        left_padding = inputs[:, -padding_size:, :]
        right_padding = inputs[:, :padding_size, :]
        padded_inputs = tf.concat([left_padding, inputs, right_padding], axis=1)
        conv_output = self.conv(padded_inputs)
        sequence_length = tf.shape(inputs)[1]
        return conv_output[:, :sequence_length, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation
        })
        return config

class AttentionMechanism(tf.keras.layers.Layer):
    def __init__(self, attention_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        self.W = Dense(attention_dim, activation='tanh')
        self.U = Dense(1, activation='linear')
    
    def call(self, inputs):
        attention_weights = self.U(self.W(inputs))
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        attended_output = tf.reduce_sum(inputs * attention_weights, axis=1)
        return attended_output, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'attention_dim': self.attention_dim})
        return config

class WeightedReconstructionLoss(tf.keras.losses.Loss):
    def __init__(self, joint_weights, temporal_weights=None, upper_body_boost=2.0, lower_body_suppress=0.2, **kwargs):
        super().__init__(**kwargs)

        adjusted_weights = np.array(joint_weights).copy()
        upper_body_indices = list(range(0, 27))
        lower_body_indices = list(range(27, 42))
        
        for idx in upper_body_indices:
            if idx < len(adjusted_weights):
                adjusted_weights[idx] *= upper_body_boost
        
        for idx in lower_body_indices:
            if idx < len(adjusted_weights):
                adjusted_weights[idx] *= lower_body_suppress

        foot_indices = list(range(36, 42))
        for idx in foot_indices:
            if idx < len(adjusted_weights):
                adjusted_weights[idx] *= 0.1
        
        self.joint_weights = tf.constant(adjusted_weights, dtype=tf.float32)
        self.temporal_weights = temporal_weights

    def call(self, y_true, y_pred):
        squared_diff = tf.square(y_true - y_pred)
        weighted_diff = squared_diff * self.joint_weights
        
        if self.temporal_weights is not None:
            temporal_weights = tf.expand_dims(self.temporal_weights, axis=-1)
            weighted_diff = weighted_diff * temporal_weights

        upper_error = tf.reduce_mean(weighted_diff[:, :, :27])
        lower_error = tf.reduce_mean(weighted_diff[:, :, 27:])
        balanced_error = upper_error * 0.85 + lower_error * 0.15
        
        return balanced_error

class MotionPeriodExtractor:
    def __init__(self):
        self.target_period = 100
    
    def extract_period(self, motion_sequence):
        try:
            wrist_data = motion_sequence[:, 12:15].flatten()
            fft_result = np.fft.fft(wrist_data)
            freqs = np.fft.fftfreq(len(wrist_data))
            
            power_spectrum = np.abs(fft_result)
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            if dominant_freq > 0:
                period = 1.0 / dominant_freq
            else:
                period = len(wrist_data)
            
            return max(period, 50)
        except:
            return self.target_period
    
    def normalize_to_standard_period(self, motion_sequence):
        current_period = self.extract_period(motion_sequence)
        scale_factor = self.target_period / current_period
        
        new_length = int(len(motion_sequence) * scale_factor)
        new_length = min(max(new_length, 50), 200)
        
        normalized_sequence = np.zeros((new_length, motion_sequence.shape[1]))
        for i in range(motion_sequence.shape[1]):
            normalized_sequence[:, i] = np.interp(
                np.linspace(0, len(motion_sequence)-1, new_length),
                np.arange(len(motion_sequence)),
                motion_sequence[:, i]
            )
        
        return normalized_sequence

class PhaseRotationProcessor:
    def __init__(self):
        self.reference_phase = 0
    
    def extract_phase(self, motion_data):
        try:
            wrist_data = motion_data[:, 12]
            analytic_signal = scipy.signal.hilbert(wrist_data)
            phase = np.angle(analytic_signal)
            return phase
        except:
            return np.zeros(len(motion_data))
    
    def rotate_to_reference_phase(self, motion_data):
        try:
            current_phase = self.extract_phase(motion_data)
            if len(current_phase) > 0:
                phase_offset = self.reference_phase - current_phase[0]
                
                rotated_data = motion_data.copy()
                for i in range(motion_data.shape[1]):
                    if i % 3 == 0:
                        if i + 1 < motion_data.shape[1]:
                            rotated_data[:, i] = motion_data[:, i] * np.cos(phase_offset) - motion_data[:, i+1] * np.sin(phase_offset)
                    elif i % 3 == 1:
                        if i > 0:
                            rotated_data[:, i] = motion_data[:, i-1] * np.sin(phase_offset) + motion_data[:, i] * np.cos(phase_offset)
                
                return rotated_data
        except:
            pass
        return motion_data

class SpectralAnalyzer:
    def __init__(self):
        self.frequency_bands = {
            'low': (0, 2),
            'mid': (2, 8),
            'high': (8, 20)
        }
    
    def extract_spectral_features(self, motion_data):
        spectral_features = []
        
        try:
            num_joints = min(motion_data.shape[1] // 3, 14)
            
            for joint_idx in range(num_joints):
                try:
                    start_idx = joint_idx * 3
                    end_idx = min(start_idx + 3, motion_data.shape[1])
                    
                    if end_idx > start_idx:
                        joint_data = motion_data[:, start_idx:end_idx]

                        if joint_data.shape[1] > 0:
                            joint_magnitude = np.linalg.norm(joint_data, axis=1)

                            if len(joint_magnitude) > 1:
                                fft_result = np.fft.fft(joint_magnitude)
                                freqs = np.fft.fftfreq(len(joint_magnitude), d=1/30)
                                power_spectrum = np.abs(fft_result)

                                for band_name, (low, high) in self.frequency_bands.items():
                                    band_mask = (freqs >= low) & (freqs <= high)
                                    if np.any(band_mask):
                                        band_power = np.mean(power_spectrum[band_mask])
                                        spectral_features.append(band_power if not np.isnan(band_power) else 0.0)
                                    else:
                                        spectral_features.append(0.0)
                            else:
                                spectral_features.extend([0.0, 0.0, 0.0])
                        else:
                            spectral_features.extend([0.0, 0.0, 0.0])
                            
                except Exception as e:
                    print(f"関節{joint_idx}のスペクトル解析エラー: {e}")
                    spectral_features.extend([0.0, 0.0, 0.0])

            while len(spectral_features) < 42:
                spectral_features.append(0.0)
                
        except Exception as e:
            print(f"スペクトル解析全体エラー: {e}")
            spectral_features = [0.0] * 42
        
        return np.array(spectral_features)

class HarmonicAnalyzer:
    def __init__(self, max_harmonics=5):
        self.max_harmonics = max_harmonics
    
    def extract_harmonics(self, motion_signal):
        try:
            fft_result = np.fft.fft(motion_signal)
            freqs = np.fft.fftfreq(len(motion_signal))

            fundamental_idx = np.argmax(np.abs(fft_result[1:len(fft_result)//2])) + 1
            fundamental_freq = freqs[fundamental_idx]

            harmonics = {}
            for h in range(1, self.max_harmonics + 1):
                harmonic_freq = fundamental_freq * h
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonics[f'H{h}'] = {
                    'amplitude': float(np.abs(fft_result[harmonic_idx])),
                    'phase': float(np.angle(fft_result[harmonic_idx]))
                }
            
            return harmonics
        except:
            return {f'H{h}': {'amplitude': 0.0, 'phase': 0.0} for h in range(1, self.max_harmonics + 1)}

class SpatioTemporalExtractor:
    def __init__(self):
        pass
    
    def extract_integrated_features(self, motion_sequence):
        features = {}
        
        try:
            features['spatial'] = {
                'joint_angles': self.calculate_joint_angles(motion_sequence),
                'velocities': np.gradient(motion_sequence, axis=0),
                'accelerations': np.gradient(np.gradient(motion_sequence, axis=0), axis=0)
            }

            features['temporal'] = {
                'period': self.extract_dominant_period(motion_sequence),
                'phase_consistency': self.calculate_phase_consistency(motion_sequence),
                'rhythm_stability': self.analyze_rhythm_stability(motion_sequence)
            }

            features['spatiotemporal'] = {
                'motion_fluidity': self.calculate_motion_fluidity(motion_sequence),
                'coordination_index': self.calculate_coordination(motion_sequence),
                'energy_efficiency': self.calculate_energy_efficiency(motion_sequence)
            }
        except:
            features = {'spatial': {}, 'temporal': {}, 'spatiotemporal': {}}
        
        return features
    
    def calculate_joint_angles(self, motion_sequence):
        try:
            angles = []
            for i in range(0, motion_sequence.shape[1]-6, 3):
                joint1 = motion_sequence[:, i:i+3]
                joint2 = motion_sequence[:, i+3:i+6]
                
                vectors = joint2 - joint1
                norms = np.linalg.norm(vectors, axis=1)
                norms[norms == 0] = 1
                
                angles.append(np.mean(norms))
            
            return np.array(angles)
        except:
            return np.zeros(13)
    
    def extract_dominant_period(self, motion_sequence):
        try:
            wrist_data = motion_sequence[:, 12:15].flatten()
            fft_result = np.fft.fft(wrist_data)
            freqs = np.fft.fftfreq(len(wrist_data))
            
            dominant_freq_idx = np.argmax(np.abs(fft_result[1:len(fft_result)//2])) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            return 1.0 / dominant_freq if dominant_freq > 0 else 100.0
        except:
            return 100.0
    
    def calculate_phase_consistency(self, motion_sequence):
        try:
            wrist_data = motion_sequence[:, 12]
            analytic_signal = scipy.signal.hilbert(wrist_data)
            phases = np.angle(analytic_signal)

            phase_diffs = np.diff(phases)
            return 1.0 / (1.0 + np.std(phase_diffs))
        except:
            return 0.5
    
    def analyze_rhythm_stability(self, motion_sequence):
        try:
            wrist_data = motion_sequence[:, 12:15]
            velocities = np.gradient(wrist_data, axis=0)
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)

            cv = np.std(velocity_magnitudes) / (np.mean(velocity_magnitudes) + 1e-8)
            return 1.0 / (1.0 + cv)
        except:
            return 0.5
    
    def calculate_motion_fluidity(self, motion_sequence):
        try:
            accelerations = np.gradient(np.gradient(motion_sequence, axis=0), axis=0)
            jerk = np.gradient(accelerations, axis=0)

            mean_jerk = np.mean(np.linalg.norm(jerk, axis=1))
            return 1.0 / (1.0 + mean_jerk)
        except:
            return 0.5
    
    def calculate_coordination(self, motion_sequence):
        try:
            upper_body = motion_sequence[:, :27]
            lower_body = motion_sequence[:, 27:]
            
            upper_velocity = np.gradient(upper_body, axis=0)
            lower_velocity = np.gradient(lower_body, axis=0)

            upper_norm = np.linalg.norm(upper_velocity, axis=1)
            lower_norm = np.linalg.norm(lower_velocity, axis=1)
            
            correlation = np.corrcoef(upper_norm, lower_norm)[0, 1]
            return max(0, correlation)
        except:
            return 0.5
    
    def calculate_energy_efficiency(self, motion_sequence):
        try:
            velocities = np.gradient(motion_sequence, axis=0)
            kinetic_energies = 0.5 * np.sum(velocities**2, axis=1)

            min_energy = np.min(kinetic_energies)
            mean_energy = np.mean(kinetic_energies)
            
            return min_energy / (mean_energy + 1e-8)
        except:
            return 0.5

class TableTennisAnomalyDetector:
    
    def __init__(self):
        self.model = None
        self.scaler_human = StandardScaler()
        self.scaler_bat = StandardScaler()
        self.scaler_features = StandardScaler()
        self.threshold = 0.0
        self.training_history = {}

        self.period_extractor = MotionPeriodExtractor()
        self.phase_processor = PhaseRotationProcessor()
        self.spectral_analyzer = SpectralAnalyzer()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.spatiotemporal_extractor = SpatioTemporalExtractor()

        self.joint_weights = self._get_enhanced_joint_weights()

        self.motion_classes = [
            'forehand_attack', 'forehand_drive', 'forehand_push',
            'backhand_attack', 'backhand_drive', 'backhand_push'
        ]

    def _get_enhanced_joint_weights(self):
        weights = np.ones(45)
        enhanced_importance = {
            0: 0.5,   # head
            1: 0.6,   # neck
            2: 2.0,   # right_shoulder
            3: 2.5,   # right_elbow
            4: 3.0,   # right_wrist
            5: 1.2,   # left_shoulder
            6: 1.0,   # left_elbow
            7: 0.8,   # left_wrist
            8: 1.5,   # spine
            9: 0.4,   # right_hip
            10: 0.2,  # right_knee
            11: 0.1,  # right_ankle
            12: 0.4,  # left_hip
            13: 0.2,  # left_knee
            14: 0.1,  # left_ankle
            15: 0.01  # right_foot_index
        }

        for joint_idx, importance in enhanced_importance.items():
            if joint_idx < 16:
                for axis in range(3):
                    weight_idx = joint_idx * 3 + axis
                    if weight_idx < len(weights):
                        weights[weight_idx] = importance

        if len(weights) >= 48:
            weights[-3:] = [2.0, 2.0, 2.0]
        
        return weights

    def extract_advanced_features(self, human_data, bat_data):
        features_list = []
        
        for human_seq, bat_seq in zip(human_data, bat_data):
            try:
                spectral_features = self.spectral_analyzer.extract_spectral_features(human_seq)

                if not isinstance(spectral_features, np.ndarray):
                    spectral_features = np.array([spectral_features] if np.isscalar(spectral_features) else spectral_features)

                try:
                    wrist_data = human_seq[:, 12] if human_seq.shape[1] > 12 else human_seq[:, 0]
                    harmonics = self.harmonic_analyzer.extract_harmonics(wrist_data)
                    harmonic_features = [harmonics[f'H{h}']['amplitude'] for h in range(1, 6)]
                except:
                    harmonic_features = [0.0] * 5

                try:
                    spatiotemporal = self.spatiotemporal_extractor.extract_integrated_features(human_seq)
                    stat_features = [
                        spatiotemporal.get('temporal', {}).get('period', 100.0),
                        spatiotemporal.get('temporal', {}).get('phase_consistency', 0.5),
                        spatiotemporal.get('temporal', {}).get('rhythm_stability', 0.5),
                        spatiotemporal.get('spatiotemporal', {}).get('motion_fluidity', 0.5),
                        spatiotemporal.get('spatiotemporal', {}).get('coordination_index', 0.5),
                        spatiotemporal.get('spatiotemporal', {}).get('energy_efficiency', 0.5)
                    ]
                except:
                    stat_features = [0.5] * 6

                try:
                    velocity_data = np.gradient(human_seq, axis=0)
                    velocity_features = np.mean(np.abs(velocity_data), axis=0)
                    
                    if np.isscalar(velocity_features):
                        velocity_features = np.array([velocity_features])
                    elif velocity_features.ndim == 0:
                        velocity_features = np.array([velocity_features.item()])

                    velocity_feature_value = velocity_features[0] if len(velocity_features) > 0 else 0.0
                    
                except Exception as e:
                    print(f"速度特徴計算エラー: {e}")
                    velocity_feature_value = 0.0

                coordination_features = []
                try:
                    max_joints = min(human_seq.shape[1] // 3, 5) 
                    for i in range(0, max_joints * 3, 3):
                        for j in range(i+3, max_joints * 3, 3):
                            if i < human_seq.shape[1] and j < human_seq.shape[1]:
                                corr_matrix = np.corrcoef(human_seq[:, i], human_seq[:, j])
                                if corr_matrix.shape == (2, 2):
                                    corr_value = corr_matrix[0, 1]
                                    if not np.isnan(corr_value):
                                        coordination_features.append(corr_value)

                    while len(coordination_features) < 5:
                        coordination_features.append(0.0)
                    coordination_features = coordination_features[:5]
                    
                except Exception as e:
                    print(f"協調特徴計算エラー: {e}")
                    coordination_features = [0.0] * 5

                combined_features = []

                spectral_safe = spectral_features.flatten()[:15] if len(spectral_features) > 0 else [0.0] * 15
                combined_features.extend(spectral_safe)

                combined_features.extend(harmonic_features)

                combined_features.extend(stat_features)

                combined_features.append(velocity_feature_value)

                combined_features.extend(coordination_features)
                
                while len(combined_features) < 32:
                    combined_features.append(0.0)
                combined_features = combined_features[:32]

                if len(combined_features) != 32:
                    print(f"警告: 特徴量長さが不正 ({len(combined_features)}), 32次元に修正")
                    combined_features = combined_features[:32] + [0.0] * max(0, 32 - len(combined_features))
                
                features_list.append(combined_features)
                
            except Exception as e:
                print(f"特徴抽出エラー（シーケンス処理中）: {e}")
                features_list.append([0.0] * 32)
        
        return np.array(features_list)

    def detect_motion_boundaries(self, pose_data, threshold=0.3):
        """動作境界検出（元の機能を保持・拡張）"""
        velocities = np.diff(pose_data, axis=0)
        weighted_velocities = velocities * self.joint_weights[:pose_data.shape[1]]
        motion_intensity = np.sum(np.abs(weighted_velocities), axis=1)

        if len(motion_intensity) > 11:
            smoothed = scipy.signal.savgol_filter(motion_intensity, 11, 3)
        else:
            smoothed = motion_intensity

        adaptive_threshold = threshold * np.std(smoothed)
        
        start_points = []
        end_points = []
        in_motion = False
        min_duration = 30
        
        for i, intensity in enumerate(smoothed):
            if not in_motion and intensity > adaptive_threshold:
                start_points.append(i)
                in_motion = True
            elif in_motion and intensity < adaptive_threshold * 0.5:
                if len(start_points) > len(end_points):
                    duration = i - start_points[-1]
                    if duration >= min_duration:
                        end_points.append(i)
                    in_motion = False
        
        if len(start_points) > len(end_points):
            end_points.append(len(smoothed) - 1)
        
        return start_points, end_points, smoothed

    def extract_complete_cycles(self, data, start_points, end_points, target_length=100):
        complete_cycles = []
        cycle_metadata = []
        
        for start, end in zip(start_points, end_points):
            if end - start < 50:
                continue
            
            original_cycle = data[start:end]

            if hasattr(self, 'use_period_normalization') and self.use_period_normalization:
                try:
                    normalized_cycle = self.period_extractor.normalize_to_standard_period(original_cycle)
                    if len(normalized_cycle) != target_length:
                        normalized_cycle = self._resample_to_fixed_length(normalized_cycle, target_length)
                except:
                    normalized_cycle = self._resample_to_fixed_length(original_cycle, target_length)
            else:
                normalized_cycle = self._resample_to_fixed_length(original_cycle, target_length)
            if hasattr(self, 'use_phase_sync') and self.use_phase_sync:
                try:
                    normalized_cycle = self.phase_processor.rotate_to_reference_phase(normalized_cycle)
                except:
                    pass 
            
            complete_cycles.append(normalized_cycle)
            cycle_metadata.append({
                'original_start': start,
                'original_end': end,
                'original_duration': end - start,
                'normalized_length': target_length
            })
        
        return np.array(complete_cycles), cycle_metadata

    def _resample_to_fixed_length(self, data, target_length):
        if len(data) == target_length:
            return data
        
        original_indices = np.linspace(0, len(data) - 1, len(data))
        target_indices = np.linspace(0, len(data) - 1, target_length)
        
        resampled_data = np.zeros((target_length, data.shape[1]))
        for i in range(data.shape[1]):
            resampled_data[:, i] = np.interp(target_indices, original_indices, data[:, i])
        
        return resampled_data

    def phase_rotation_augmentation(self, data, n_rotations=6):
        augmented_data = []
        
        for sequence in data:
            for rotation in range(n_rotations):
                rotation_angle = 2 * np.pi * rotation / n_rotations
                
                rotated_sequence = sequence.copy()
                for i in range(0, sequence.shape[1], 3):
                    if i + 2 < sequence.shape[1]:
                        x, y = sequence[:, i], sequence[:, i+1]
                        rotated_sequence[:, i] = x * np.cos(rotation_angle) - y * np.sin(rotation_angle)
                        rotated_sequence[:, i+1] = x * np.sin(rotation_angle) + y * np.cos(rotation_angle)
                
                augmented_data.append(rotated_sequence)
        
        return np.array(augmented_data)

    def build_enhanced_autoencoder(self, input_shape, latent_dim=32, use_attention=True):
        human_input = Input(shape=(input_shape[0], 42), name='human_input')
        bat_input = Input(shape=(input_shape[0], 3), name='bat_input')
        features_input = Input(shape=(32,), name='features_input')

        h1 = CircularConv1D(64, 5, activation='relu')(human_input)
        h1 = BatchNormalization()(h1)
        h1 = Dropout(0.3)(h1)
        
        h2 = CircularConv1D(32, 3, activation='relu')(h1)
        h2 = BatchNormalization()(h2)
        h2 = Dropout(0.3)(h2)
        
        h3 = LSTM(64, return_sequences=True if use_attention else False)(h2)
        
        if use_attention:
            human_attended, self.human_attention_weights = AttentionMechanism(64)(h3)
        else:
            human_attended = h3
            self.human_attention_weights = None

        b1 = CircularConv1D(32, 5, activation='relu')(bat_input)
        b1 = BatchNormalization()(b1)
        b1 = Dropout(0.2)(b1)
        
        b2 = LSTM(32, return_sequences=True if use_attention else False)(b1)
        
        if use_attention:
            bat_attended, self.bat_attention_weights = AttentionMechanism(32)(b2)
        else:
            bat_attended = b2
            self.bat_attention_weights = None

        f1 = Dense(64, activation='relu')(features_input)
        f1 = BatchNormalization()(f1)
        f1 = Dropout(0.2)(f1)
        f1 = Dense(32, activation='relu')(f1)
        features_encoded = Dense(16, activation='relu')(f1)

        if use_attention:
            combined = Concatenate()([human_attended, bat_attended, features_encoded])
        else:
            combined = Concatenate()([human_attended, bat_attended, features_encoded])
        
        latent = Dense(latent_dim, activation='relu', name='latent')(combined)

        decoded = Dense(112, activation='relu')(latent)
        decoded = RepeatVector(input_shape[0])(decoded)
        decoded = LSTM(112, return_sequences=True)(decoded)
        decoded = LSTM(80, return_sequences=True)(decoded)

        human_output = TimeDistributed(Dense(42, activation='linear'), name='human_reconstructed')(decoded)
        bat_output = TimeDistributed(Dense(3, activation='linear'), name='bat_reconstructed')(decoded)

        features_decoded = Dense(64, activation='relu')(latent)
        features_output = Dense(32, activation='linear', name='features_reconstructed')(features_decoded)

        outputs = [human_output, bat_output, features_output]
        
        autoencoder = Model(
            inputs=[human_input, bat_input, features_input],
            outputs=outputs
        )
        
        return autoencoder

    def train_model(self, human_data, bat_data, motion_type=0,
                    validation_split=0.2, epochs=200, batch_size=32,
                    use_advanced_features=True):
        combined_data = np.concatenate([human_data, bat_data], axis=2)
        boundaries = []
        motion_intensities = []
        
        for i, sequence in enumerate(combined_data):
            start_points, end_points, intensity = self.detect_motion_boundaries(sequence)
            boundaries.append((start_points, end_points))
            motion_intensities.append(intensity)

        all_cycles_human = []
        all_cycles_bat = []
        
        for i, (sequence_human, sequence_bat) in enumerate(zip(human_data, bat_data)):
            start_points, end_points = boundaries[i]
            if start_points and end_points:
                cycles_human, _ = self.extract_complete_cycles(sequence_human, start_points, end_points)
                cycles_bat, _ = self.extract_complete_cycles(sequence_bat, start_points, end_points)
                
                if len(cycles_human) > 0 and len(cycles_bat) > 0:
                    all_cycles_human.extend(cycles_human)
                    all_cycles_bat.extend(cycles_bat)
        
        if len(all_cycles_human) == 0:
            raise ValueError("有効な動作周期が抽出できませんでした")
        
        all_cycles_human = np.array(all_cycles_human)
        all_cycles_bat = np.array(all_cycles_bat)

        if use_advanced_features:
            advanced_features = self.extract_advanced_features(all_cycles_human, all_cycles_bat)
            print(f"   抽出された特徴次元: {advanced_features.shape}")
        else:
            advanced_features = np.zeros((len(all_cycles_human), 32))

        n_rotations = getattr(self, 'n_rotations', 6)
        augmented_human = self.phase_rotation_augmentation(all_cycles_human, n_rotations)
        augmented_bat = self.phase_rotation_augmentation(all_cycles_bat, n_rotations)

        augmented_features = np.tile(advanced_features, (n_rotations, 1))
        
        print(f"   拡張後のデータ数: {len(augmented_human)}")

        human_reshaped = augmented_human.reshape(-1, augmented_human.shape[-1])
        bat_reshaped = augmented_bat.reshape(-1, augmented_bat.shape[-1])
        
        human_normalized = self.scaler_human.fit_transform(human_reshaped)
        bat_normalized = self.scaler_bat.fit_transform(bat_reshaped)
        features_normalized = self.scaler_features.fit_transform(augmented_features)
        
        human_normalized = human_normalized.reshape(augmented_human.shape)
        bat_normalized = bat_normalized.reshape(augmented_bat.shape)

        latent_dim = getattr(self, 'latent_dim', 32)  # デフォルト32
        
        input_shape = (100, 42)
        self.model = self.build_enhanced_autoencoder(input_shape, latent_dim=latent_dim, use_attention=True)

        human_weights = self.joint_weights[:42]
        bat_weights = np.ones(3) * 2.0
        
        upper_body_boost = getattr(self, 'upper_body_boost', 2.0)
        lower_body_suppress = getattr(self, 'lower_body_suppress', 0.2)
        
        weighted_loss_human = WeightedReconstructionLoss(
            human_weights, 
            upper_body_boost=upper_body_boost,
            lower_body_suppress=lower_body_suppress
        )
        weighted_loss_bat = WeightedReconstructionLoss(bat_weights)

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'human_reconstructed': weighted_loss_human,
                'bat_reconstructed': weighted_loss_bat,
                'features_reconstructed': 'mse'
            },
            loss_weights={
                'human_reconstructed': 0.6,
                'bat_reconstructed': 0.2,
                'features_reconstructed': 0.2
            },
            metrics={
                'human_reconstructed': ['mae'],
                'bat_reconstructed': ['mae'],
                'features_reconstructed': ['mae']
            }
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7)
        ]

        outputs = [human_normalized, bat_normalized, features_normalized]
        
        self.training_history = self.model.fit(
            [human_normalized, bat_normalized, features_normalized],
            outputs,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        predictions = self.model.predict([human_normalized, bat_normalized, features_normalized])
        
        human_errors = np.mean((human_normalized - predictions[0])**2, axis=(1,2))
        bat_errors = np.mean((bat_normalized - predictions[1])**2, axis=(1,2))
        features_errors = np.mean((features_normalized - predictions[2])**2, axis=1)

        total_errors = human_errors * 0.6 + bat_errors * 0.2 + features_errors * 0.2

        self.threshold = np.percentile(total_errors, 99)
        
        return {
            'training_loss': self.training_history.history['loss'],
            'validation_loss': self.training_history.history['val_loss'],
            'threshold': self.threshold,
            'total_cycles': len(all_cycles_human),
            'augmented_samples': len(augmented_human),
            'features_shape': advanced_features.shape
        }

    def predict_anomaly(self, human_data, bat_data):
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        advanced_features = self.extract_advanced_features(human_data, bat_data)

        human_reshaped = human_data.reshape(-1, human_data.shape[-1])
        bat_reshaped = bat_data.reshape(-1, bat_data.shape[-1])
        
        human_normalized = self.scaler_human.transform(human_reshaped).reshape(human_data.shape)
        bat_normalized = self.scaler_bat.transform(bat_reshaped).reshape(bat_data.shape)
        features_normalized = self.scaler_features.transform(advanced_features)

        predictions = self.model.predict([human_normalized, bat_normalized, features_normalized])

        human_errors = np.mean((human_normalized - predictions[0])**2, axis=2)
        bat_errors = np.mean((bat_normalized - predictions[1])**2, axis=2)
        features_errors = np.mean((features_normalized - predictions[2])**2, axis=1)

        total_errors = (np.mean(human_errors, axis=1) * 0.6 + 
                       np.mean(bat_errors, axis=1) * 0.2 + 
                       features_errors * 0.2)

        is_anomaly = total_errors > self.threshold

        explanations = {
            'component_errors': {
                'human': np.mean(human_errors, axis=1),
                'bat': np.mean(bat_errors, axis=1),
                'features': features_errors
            },
            'attention_weights': None
        }

        if hasattr(self, 'human_attention_weights') and self.human_attention_weights is not None:
            explanations['attention_weights'] = {
                'human': self.human_attention_weights,
                'bat': self.bat_attention_weights if hasattr(self, 'bat_attention_weights') else None
            }
        
        return total_errors, is_anomaly, explanations

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("保存するモデルがありません")

        model_path = filepath.replace('.pkl', '_enhanced_model.h5')

        self.model.save(model_path, save_format='h5')

        enhanced_config = {
            'scaler_human': self.scaler_human,
            'scaler_bat': self.scaler_bat,
            'scaler_features': self.scaler_features,
            'threshold': self.threshold,
            'joint_weights': self.joint_weights,
            'training_history': self.training_history.history if self.training_history else None,
            'model_architecture': 'enhanced_multimodal',
            'features_enabled': True,
            'attention_enabled': True,
            'custom_objects': ['CircularConv1D', 'AttentionMechanism', 'WeightedReconstructionLoss']
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(enhanced_config, f)
        
        return model_path, filepath

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            config = pickle.load(f)
        
        self.scaler_human = config.get('scaler_human', StandardScaler())
        self.scaler_bat = config.get('scaler_bat', StandardScaler())
        self.scaler_features = config.get('scaler_features', StandardScaler())
        self.threshold = config.get('threshold', 0.0)
        self.joint_weights = config.get('joint_weights', self._get_enhanced_joint_weights())

        model_path = filepath.replace('.pkl', '_enhanced_model.h5')
        if os.path.exists(model_path):
            custom_objects = {
                'CircularConv1D': CircularConv1D,
                'AttentionMechanism': AttentionMechanism,
                'WeightedReconstructionLoss': WeightedReconstructionLoss
            }
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        
        return True

class TableTennisDataProcessor:
    @staticmethod
    def load_ttmd6_data(human_folder, bat_folder, target_motion=None):
        human_files = [f for f in os.listdir(human_folder) if f.endswith('.csv')]
        bat_files = [f for f in os.listdir(bat_folder) if f.endswith('.csv')]
        
        matched_pairs = []
        for human_file in human_files:
            parts = human_file.replace('.csv', '').split('_')
            if len(parts) >= 4:
                label = int(parts[3])
                
                if target_motion is not None and label != target_motion:
                    continue
                
                for bat_file in bat_files:
                    bat_parts = bat_file.replace('.csv', '').split('_')
                    if len(bat_parts) >= 4 and int(bat_parts[3]) == label:
                        matched_pairs.append((human_file, bat_file, label))
                        break
        
        human_data = []
        bat_data = []
        labels = []
        
        for human_file, bat_file, label in matched_pairs:
            try:
                human_df = pd.read_csv(os.path.join(human_folder, human_file))
                bat_df = pd.read_csv(os.path.join(bat_folder, bat_file))
                
                human_values = human_df.select_dtypes(include=[np.number]).values
                bat_values = bat_df.select_dtypes(include=[np.number]).values
                
                min_length = min(len(human_values), len(bat_values))
                human_values = human_values[:min_length]
                bat_values = bat_values[:min_length]
                
                human_data.append(human_values)
                bat_data.append(bat_values)
                labels.append(label)
                
            except Exception as e:
                print(f"ファイル読み込みエラー: {human_file}, {bat_file} - {e}")
        
        return np.array(human_data), np.array(bat_data), np.array(labels)

class TableTennisGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("卓球動作異常検出システム")
        self.root.geometry("1500x1000")

        self.detector = TableTennisAnomalyDetector()
        self.processor = TableTennisDataProcessor()

        self.human_data = None
        self.bat_data = None
        self.labels = None
        self.training_results = None

        self.create_gui()

    def create_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        self.create_data_tab(notebook)
        self.create_training_tab(notebook)
        self.create_results_tab(notebook)
        self.create_advanced_features_tab(notebook)

        self.create_control_panel(main_frame)

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def create_data_tab(self, notebook):
        data_frame = ttk.Frame(notebook, padding="10")
        notebook.add(data_frame, text="データ")

        file_frame = ttk.LabelFrame(data_frame, text="ファイル選択", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(file_frame, text="Humanフォルダ:").grid(row=0, column=0, sticky=tk.W)
        self.human_folder_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.human_folder_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="参照", command=self.select_human_folder).grid(row=0, column=2)
        
        ttk.Label(file_frame, text="Batフォルダ:").grid(row=1, column=0, sticky=tk.W)
        self.bat_folder_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.bat_folder_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="参照", command=self.select_bat_folder).grid(row=1, column=2)

        ttk.Button(file_frame, text="データ読み込み", command=self.load_data).grid(row=2, column=0, columnspan=3, pady=10)

        motion_frame = ttk.LabelFrame(data_frame, text="動作タイプ選択", padding="10")
        motion_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.motion_type_var = tk.IntVar(value=-1)
        ttk.Radiobutton(motion_frame, text="全動作", variable=self.motion_type_var, value=-1).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(motion_frame, text="フォアハンドアタック", variable=self.motion_type_var, value=0).grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(motion_frame, text="フォアハンドドライブ", variable=self.motion_type_var, value=1).grid(row=0, column=2, sticky=tk.W)
        ttk.Radiobutton(motion_frame, text="フォアハンドプッシュ", variable=self.motion_type_var, value=2).grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(motion_frame, text="バックハンドアタック", variable=self.motion_type_var, value=3).grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(motion_frame, text="バックハンドドライブ", variable=self.motion_type_var, value=4).grid(row=1, column=2, sticky=tk.W)
        ttk.Radiobutton(motion_frame, text="バックハンドプッシュ", variable=self.motion_type_var, value=5).grid(row=2, column=0, sticky=tk.W)

        info_frame = ttk.LabelFrame(data_frame, text="データ情報", padding="10")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.data_info_text = tk.Text(info_frame, height=15, width=80)
        self.data_info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.data_info_text.yview)
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.data_info_text.configure(yscrollcommand=info_scrollbar.set)

        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(2, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)

    def create_training_tab(self, notebook):
        training_frame = ttk.Frame(notebook, padding="10")
        notebook.add(training_frame, text="訓練設定")
        
        # 基本設定（元の機能を保持）
        basic_frame = ttk.LabelFrame(training_frame, text="設定", padding="10")
        basic_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(basic_frame, text="エポック数:").grid(row=0, column=0, sticky=tk.W)
        self.epochs_var = tk.IntVar(value=200)
        ttk.Spinbox(basic_frame, from_=50, to=1000, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(basic_frame, text="バッチサイズ:").grid(row=1, column=0, sticky=tk.W)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Spinbox(basic_frame, from_=8, to=128, textvariable=self.batch_size_var, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(basic_frame, text="検証データ比率:").grid(row=2, column=0, sticky=tk.W)
        self.validation_split_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(basic_frame, from_=0.1, to=0.5, increment=0.05,
                   textvariable=self.validation_split_var, width=10).grid(row=2, column=1, padx=5)

        advanced_frame = ttk.LabelFrame(training_frame, text="設定", padding="10")
        advanced_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(advanced_frame, text="潜在空間次元:").grid(row=0, column=0, sticky=tk.W)
        self.latent_dim_var = tk.IntVar(value=32)
        ttk.Spinbox(advanced_frame, from_=16, to=128, textvariable=self.latent_dim_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(advanced_frame, text="位相回転数:").grid(row=1, column=0, sticky=tk.W)
        self.n_rotations_var = tk.IntVar(value=6)
        ttk.Spinbox(advanced_frame, from_=4, to=12, textvariable=self.n_rotations_var, width=10).grid(row=1, column=1, padx=5)

        ttk.Label(advanced_frame, text="上半身強化倍率:").grid(row=2, column=0, sticky=tk.W)
        self.upper_body_boost_var = tk.DoubleVar(value=2.0)
        ttk.Scale(advanced_frame, from_=1.0, to=5.0, variable=self.upper_body_boost_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=2, column=1, padx=5)
        
        ttk.Label(advanced_frame, text="下半身抑制倍率:").grid(row=3, column=0, sticky=tk.W)
        self.lower_body_suppress_var = tk.DoubleVar(value=0.2)
        ttk.Scale(advanced_frame, from_=0.1, to=1.0, variable=self.lower_body_suppress_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=3, column=1, padx=5)

        tech_frame = ttk.LabelFrame(training_frame, text="機能", padding="10")
        tech_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.use_advanced_features = tk.BooleanVar(value=True)
        ttk.Checkbutton(tech_frame, text="特徴抽出",
                       variable=self.use_advanced_features).grid(row=0, column=0, sticky=tk.W)
        
        self.use_attention_mechanism = tk.BooleanVar(value=True)
        ttk.Checkbutton(tech_frame, text="注意機構",
                       variable=self.use_attention_mechanism).grid(row=1, column=0, sticky=tk.W)
        
        self.use_period_normalization = tk.BooleanVar(value=True)
        ttk.Checkbutton(tech_frame, text="動作周期正規化",
                       variable=self.use_period_normalization).grid(row=2, column=0, sticky=tk.W)
        
        self.use_phase_sync = tk.BooleanVar(value=True)
        ttk.Checkbutton(tech_frame, text="位相同期",
                       variable=self.use_phase_sync).grid(row=3, column=0, sticky=tk.W)

        log_frame = ttk.LabelFrame(training_frame, text="訓練ログ", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.training_log = tk.Text(log_frame, height=12, width=80)
        self.training_log.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.training_log.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.training_log.configure(yscrollcommand=log_scrollbar.set)

        training_frame.columnconfigure(0, weight=1)
        training_frame.rowconfigure(3, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

    def create_results_tab(self, notebook):
        results_frame = ttk.Frame(notebook, padding="10")
        notebook.add(results_frame, text="結果")

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, results_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        results_info_frame = ttk.LabelFrame(results_frame, text="結果情報", padding="10")
        results_info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = tk.Text(results_info_frame, height=10, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        results_scrollbar = ttk.Scrollbar(results_info_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=results_scrollbar.set)

        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        results_info_frame.columnconfigure(0, weight=1)
        results_info_frame.rowconfigure(0, weight=1)

    def create_advanced_features_tab(self, notebook):
        advanced_frame = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_frame, text="機能")

        control_frame = ttk.LabelFrame(advanced_frame, text="技術分析制御", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(control_frame, text="注意重み分析", 
                  command=self.analyze_attention_weights).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="特徴重要度分析", 
                  command=self.analyze_feature_importance).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="異常要因分析", 
                  command=self.analyze_anomaly_factors).grid(row=0, column=2, padx=5, pady=5)

        analysis_frame = ttk.LabelFrame(advanced_frame, text="分析結果", padding="10")
        analysis_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.analysis_text = tk.Text(analysis_frame, height=15, width=80)
        self.analysis_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        analysis_scrollbar = ttk.Scrollbar(analysis_frame, orient=tk.VERTICAL, command=self.analysis_text.yview)
        analysis_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.analysis_text.configure(yscrollcommand=analysis_scrollbar.set)

        advanced_frame.columnconfigure(0, weight=1)
        advanced_frame.rowconfigure(2, weight=1)
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(0, weight=1)

    def create_control_panel(self, main_frame):
        control_frame = ttk.LabelFrame(main_frame, text="制御", padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Button(control_frame, text="訓練開始", command=self.start_training).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="モデル保存", command=self.save_model).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="モデル読み込み", command=self.load_model).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="異常検出テスト", command=self.test_anomaly_detection).grid(row=0, column=3, padx=5)

        self.status_var = tk.StringVar(value="準備完了")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=1, column=0, columnspan=4, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)

    def select_human_folder(self):
        folder = filedialog.askdirectory(title="Humanフォルダを選択")
        if folder:
            self.human_folder_var.set(folder)

    def select_bat_folder(self):
        folder = filedialog.askdirectory(title="Batフォルダを選択")
        if folder:
            self.bat_folder_var.set(folder)

    def load_data(self):
        human_folder = self.human_folder_var.get()
        bat_folder = self.bat_folder_var.get()
        
        if not human_folder or not bat_folder:
            messagebox.showerror("エラー", "Human、Batフォルダを両方選択してください")
            return
        
        if not os.path.exists(human_folder) or not os.path.exists(bat_folder):
            messagebox.showerror("エラー", "選択されたフォルダが存在しません")
            return
        
        try:
            self.status_var.set("データ読み込み中...")
            self.progress_var.set(25)
            self.root.update()
            
            motion_type = self.motion_type_var.get()
            target_motion = None if motion_type == -1 else motion_type
            
            self.human_data, self.bat_data, self.labels = self.processor.load_ttmd6_data(
                human_folder, bat_folder, target_motion
            )
            
            self.progress_var.set(100)
            self.status_var.set("データ読み込み完了")
            
            # データ情報表示
            info_text = f"""
拡張データ読み込み完了！

読み込み統計:
- シーケンス数: {len(self.human_data)}
- Human データ形状: {self.human_data.shape}  
- Bat データ形状: {self.bat_data.shape}
- ラベル数: {len(self.labels)}
"""

            unique_labels, counts = np.unique(self.labels, return_counts=True)
            motion_names = ['フォアハンドアタック', 'フォアハンドドライブ', 'フォアハンドプッシュ',
                           'バックハンドアタック', 'バックハンドドライブ', 'バックハンドプッシュ']
            
            for label, count in zip(unique_labels, counts):
                if 0 <= label < len(motion_names):
                    info_text += f"- {motion_names[label]}: {count}個\n"
                else:
                    info_text += f"- 動作タイプ{label}: {count}個\n"
            
            info_text += "訓練を開始できます"
            
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, info_text)
            
            messagebox.showinfo("完了", f"{len(self.human_data)}個のシーケンスを読み込みました")
            
        except Exception as e:
            self.status_var.set("データ読み込み失敗")
            messagebox.showerror("エラー", f"データ読み込みエラー: {str(e)}")

    def start_training(self):
        if self.human_data is None or self.bat_data is None:
            messagebox.showerror("エラー", "先にデータを読み込んでください")
            return
        self.detector.use_period_normalization = self.use_period_normalization.get()
        self.detector.use_phase_sync = self.use_phase_sync.get()
        self.detector.n_rotations = self.n_rotations_var.get()
        self.detector.upper_body_boost = self.upper_body_boost_var.get()
        self.detector.lower_body_suppress = self.lower_body_suppress_var.get()
        self.detector.latent_dim = self.latent_dim_var.get()
        training_thread = threading.Thread(target=self._run_training)
        training_thread.daemon = True
        training_thread.start()

    def _run_training(self):
        try:
            self.status_var.set("訓練プロセス開始...")
            self.progress_var.set(0)

            print(f"Human data shape: {self.human_data.shape}")
            print(f"Bat data shape: {self.bat_data.shape}")

            if len(self.human_data) < 3:
                raise ValueError("訓練には最低3個のサンプルが必要です")

            if self.human_data.shape[-1] < 42:
                print(f"警告: Human データ次元が不足 ({self.human_data.shape[-1]})")
            
            if self.bat_data.shape[-1] < 3:
                print(f"警告: Bat データ次元が不足 ({self.bat_data.shape[-1]})")

            self.training_log.delete(1.0, tk.END)
            
            def update_log(message):
                self.training_log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
                self.training_log.see(tk.END)
                self.root.update()

            self.training_results = self.detector.train_model(
                self.human_data,
                self.bat_data,
                validation_split=self.validation_split_var.get(),
                epochs=self.epochs_var.get(),
                batch_size=self.batch_size_var.get(),
                use_advanced_features=self.use_advanced_features.get()
            )

            self.root.after(0, self.update_results_display)
            self.status_var.set("訓練完了")
            self.progress_var.set(100)
            
        except Exception as e:
            error_msg = f"訓練エラー: {str(e)}"
            print(f"詳細エラー: {e}")
            import traceback
            traceback.print_exc()
            
            self.status_var.set(error_msg)
            messagebox.showerror("訓練エラー", error_msg)

    def update_results_display(self):
        if self.training_results is None:
            return

        self.ax1.clear()
        self.ax2.clear()

        if 'training_loss' in self.training_results and 'validation_loss' in self.training_results:
            epochs = range(1, len(self.training_results['training_loss']) + 1)
            self.ax1.plot(epochs, self.training_results['training_loss'], 'b-', label='Training Loss', linewidth=2)
            self.ax1.plot(epochs, self.training_results['validation_loss'], 'r-', label='Validation Loss', linewidth=2)
            self.ax1.set_title('訓練履歴')
            self.ax1.set_xlabel('エポック')
            self.ax1.set_ylabel('損失')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)

        if 'threshold' in self.training_results:
            threshold_line = [self.training_results['threshold']] * 50
            self.ax2.plot(threshold_line, 'g--', linewidth=2, label=f'異常検出閾値: {self.training_results["threshold"]:.6f}')
            self.ax2.set_title('異常検出閾値（Human60%+Bat20%+Features20%）')
            self.ax2.set_ylabel('閾値')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
        
        self.canvas.draw()

        results_info = f"""
===  訓練結果 ===

基本統計:
- 総周期数: {self.training_results.get('total_cycles', 'N/A')}
- 拡張サンプル数: {self.training_results.get('augmented_samples', 'N/A')}
- 異常検出閾値: {self.training_results.get('threshold', 'N/A'):.6f}

特徴:
- 特徴形状: {self.training_results.get('features_shape', 'N/A')}
"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_info)

    def save_model(self):
        if self.detector.model is None:
            messagebox.showerror("エラー", "保存するモデルがありません")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="モデル保存"
        )
        
        if file_path:
            try:
                model_path, config_path = self.detector.save_model(file_path)
                messagebox.showinfo("完了", f"モデルを保存しました:\n{model_path}\n{config_path}")
            except Exception as e:
                messagebox.showerror("エラー", f"モデル保存エラー: {str(e)}")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="モデル読み込み"
        )
        
        if file_path:
            try:
                self.detector.load_model(file_path)
                messagebox.showinfo("完了", "モデルを読み込みました")
                self.status_var.set("モデル読み込み完了")
            except Exception as e:
                messagebox.showerror("エラー", f"モデル読み込みエラー: {str(e)}")

    def test_anomaly_detection(self):
        if self.detector.model is None:
            messagebox.showerror("エラー", "先にモデルを訓練または読み込んでください")
            return
        
        if self.human_data is None or self.bat_data is None:
            messagebox.showerror("エラー", "テストデータがありません")
            return
        
        try:
            test_indices = np.random.choice(len(self.human_data), min(10, len(self.human_data)), replace=False)
            test_human = self.human_data[test_indices]
            test_bat = self.bat_data[test_indices]
            anomaly_scores, predictions, explanations = self.detector.predict_anomaly(test_human, test_bat)

            test_results = "=== 異常検出テスト結果 ===\n\n"
            
            for i, (score, is_anomaly) in enumerate(zip(anomaly_scores, predictions)):
                status = "異常" if is_anomaly else "正常"
                test_results += f"サンプル {i+1}: {status} (統合スコア: {score:.6f})\n"
）
                if explanations and 'component_errors' in explanations:
                    comp_errors = explanations['component_errors']
                    test_results += f"  - Human誤差: {comp_errors['human'][i]:.6f} (60%重み)\n"
                    test_results += f"  - Bat誤差: {comp_errors['bat'][i]:.6f} (20%重み)\n"
                    test_results += f"  - Features誤差: {comp_errors['features'][i]:.6f} (20%重み)\n"

                if explanations and 'attention_weights' in explanations and explanations['attention_weights']:
                    test_results += f"  - 注意機構: 解釈可能な重要フレーム検出\n"
                
                test_results += "\n"
            
            test_results += f"\n統合異常検出閾値: {self.detector.threshold:.6f}\n"
            test_results += f"異常検出数: {np.sum(predictions)}/{len(predictions)}\n"
            test_results += f"検出率: {np.sum(predictions)/len(predictions)*100:.1f}%\n\n"

            result_window = tk.Toplevel(self.root)
            result_window.title("異常検出テスト結果")
            result_window.geometry("700x600")
            
            result_text = tk.Text(result_window, wrap=tk.WORD)
            result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            result_text.insert(tk.END, test_results)
            
        except Exception as e:
            messagebox.showerror("エラー", f"異常検出テストエラー: {str(e)}")
    def analyze_attention_weights(self):
        if not hasattr(self.detector, 'model') or self.detector.model is None:
            messagebox.showerror("エラー", "先にモデルを訓練してください")
            return

    def analyze_feature_importance(self):
        pass

    def analyze_anomaly_factors(self):
        pass

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = TableTennisGUI()
        app.run()
    except Exception as e:
        print(f"システム起動エラー: {e}")
        import traceback
        traceback.print_exc()
