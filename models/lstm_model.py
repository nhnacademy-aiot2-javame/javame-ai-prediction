#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM 기반 예측 유지보수 모델

두 가지 주요 모델 구현:
1. FailureLSTM: 설비 고장 예측 모델
2. ResourceLSTM: 자원 사용량 예측 및 용량 증설 계획 모델
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import seaborn as sns
import json
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# 공통 유틸리티 함수
def create_sequence_dataset(data, input_window, pred_horizon=1):
    """
    시계열 데이터를 입력-출력 시퀀스 쌍으로 변환
    
    Args:
        data (np.array): 변환할 데이터
        input_window (int): 입력 시퀀스 길이
        pred_horizon (int): 예측 지평선(출력 길이)
        
    Returns:
        np.array: X 시퀀스 (입력)
        np.array: y 시퀀스 (출력)
    """
    X, y = [], []
    for i in range(len(data) - input_window - pred_horizon + 1):
        X.append(data[i:i + input_window])
        y.append(data[i + input_window:i + input_window + pred_horizon])
    
    return np.array(X), np.array(y)

# 기본 모델 클래스 (인터페이스)
class BaseTimeSeriesModel:
    """시계열 예측 모델을 위한 기본 인터페이스"""
    
    def __init__(self, input_dim, input_window, pred_horizon=1, model_save_path=None):
        self.input_dim = input_dim
        self.input_window = input_window
        self.pred_horizon = pred_horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        self.last_training_time = None
        self.model_save_path = model_save_path
    
    def _build_model(self):
        """모델 아키텍처 구축 (서브클래스에서 구현)"""
        raise NotImplementedError("서브클래스에서 구현해야 합니다")
    
    def fit(self, df, target_col, **kwargs):
        """모델 학습 (서브클래스에서 구현)"""
        raise NotImplementedError("서브클래스에서 구현해야 합니다")
    
    def predict(self, df, target_col=None):
        """예측 수행 (서브클래스에서 구현)"""
        raise NotImplementedError("서브클래스에서 구현해야 합니다")
    
    def plot_results(self, df, target_col, results, save_dir=None):
        """결과 시각화 (서브클래스에서 구현)"""
        raise NotImplementedError("서브클래스에서 구현해야 합니다")
    
    def get_training_info(self):
        """학습 정보 반환 (서브클래스에서 구현)"""
        raise NotImplementedError("서브클래스에서 구현해야 합니다")

    def save_model(self, path=None):
        """모델 저장"""
        save_path = path or self.model_save_path
        if save_path and self.model is not None:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            logger.info(f"모델 저장 완료: {save_path}")
            return True
        return False
    
    def load_model(self, path=None):
        """모델 로드"""
        load_path = path or self.model_save_path
        if load_path and os.path.exists(load_path):
            self.model = tf.keras.models.load_model(load_path)
            logger.info(f"모델 로드 완료: {load_path}")
            return True
        return False

# =============================================================
# 1. LSTM 기반 고장 예측 모델
# =============================================================

class FailureLSTM(BaseTimeSeriesModel):
    def __init__(self, input_dim, input_window=24, pred_horizon=1, lstm_units=64, dropout_rate=0.2, model_save_path='model_weights/failure_lstm.h5'):
        """
        LSTM 기반 고장 예측 모델
        
        Args:
            input_dim (int): 입력 특성 수
            input_window (int): 입력 시퀀스 길이
            pred_horizon (int): 예측 지평선
            lstm_units (int): LSTM 유닛 수
            dropout_rate (float): 드롭아웃 비율
            model_save_path (str): 모델 저장 경로
        """
        super().__init__(input_dim, input_window, pred_horizon, model_save_path)
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
    
    def _build_model(self):
        """LSTM 기반 고장 예측 모델 구축"""
        model = Sequential([
            # 입력층: (batch_size, time_steps, features)
            LSTM(self.lstm_units, activation='tanh', return_sequences=True, 
                 input_shape=(self.input_window, self.input_dim)),
            Dropout(self.dropout_rate),
            
            # 두 번째 LSTM 층
            LSTM(self.lstm_units//2, activation='tanh'),
            Dropout(self.dropout_rate),
            
            # 출력층 (고장 확률)
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=BinaryCrossentropy(),
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def fit(self, df, target_col, failure_threshold=None, epochs=100, batch_size=32, validation_split=0.2):
        """
        모델 학습
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            target_col (str): 예측할 타겟 열 이름
            failure_threshold (float): 고장 임계값 (None이면 자동 계산)
            epochs (int): 학습 에포크 수
            batch_size (int): 배치 크기
            validation_split (float): 검증 데이터 비율
            
        Returns:
            history: 학습 이력
        """
        # 학습 시작 시간 기록
        self.last_training_time = datetime.now()
        
        # 특성과 타겟 분리
        features = df.values
        
        # 타겟 열 인덱스 찾기
        target_col_idx = df.columns.get_loc(target_col)
        
        # 데이터 정규화
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        
        # 타겟 값 정규화 확인
        target_values = scaled_features[:, target_col_idx]
        
        # 임계값 자동 계산 (필요한 경우)
        if failure_threshold is None:
            # 75 퍼센타일을 임계값으로 사용
            failure_threshold = np.percentile(target_values, 75)
            logger.info(f"임계값 자동 설정: {failure_threshold}")
        
        # 데이터가 충분한지 확인
        required_samples = self.input_window + self.pred_horizon
        if len(scaled_features) < required_samples:
            logger.warning(f"데이터가 충분하지 않습니다. 최소 {required_samples}개 필요, 현재 {len(scaled_features)}개")
            self.history = None
            return None
        
        # 시퀀스 데이터 생성
        X_seqs = []
        y_vals = []
        total = len(scaled_features)
        
        for i in range(total - self.input_window - self.pred_horizon + 1):
            seq = scaled_features[i:i + self.input_window]
            idx = i + self.input_window + self.pred_horizon - 1
            X_seqs.append(seq)
            y_vals.append(scaled_features[idx, target_col_idx])

        X = np.array(X_seqs)
        y_vals = np.array(y_vals)
        
        # 임계값 기준 이진 레이블 생성
        y_binary = (y_vals > failure_threshold).astype(int)
        
        # 클래스 불균형 확인
        pos_ratio = np.mean(y_binary)
        logger.info(f"정상/고장 비율: {1-pos_ratio:.2f}/{pos_ratio:.2f}")
        
        # 클래스 가중치 계산 (불균형 처리)
        if pos_ratio > 0 and pos_ratio < 1:
            class_weight = {0: pos_ratio, 1: 1 - pos_ratio}
        else:
            class_weight = None
        
        # 학습/검증 데이터 분할
        n_samples = len(X)
        if n_samples == 1:
            logger.warning("샘플이 1개뿐이라 검증을 건너뜁니다.")
            X_train, y_train = X, y_binary
            X_val, y_val = None, None
            val_data = None
        else:
            train_size = max(int(len(X) * (1 - validation_split)), 1)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y_binary[:train_size], y_binary[train_size:]
            val_data = (X_val, y_val) if len(X_val) > 0 else None

        # 배치 크기 조정
        n_train = len(X_train)
        eff_bs = max(1, min(batch_size, n_train))
        if eff_bs != batch_size:
            logger.warning(f"batch_size({batch_size})를 train 샘플수({n_train})에 맞춰 {eff_bs}로 조정합니다.")

        # 콜백 설정
        callbacks = []
        if val_data is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # 모델 체크포인트
        if self.model_save_path:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            
            checkpoint = ModelCheckpoint(
                self.model_save_path,
                monitor='val_loss' if val_data else 'loss',
                save_best_only=True,
                mode='min'
            )
            callbacks.append(checkpoint)
        
        # 모델 학습
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=eff_bs,
            validation_data=val_data,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # 모델 저장
        if self.model_save_path:
            self.save_model()
        
        return self.history
    
    def predict(self, df):
        """
        고장 확률 예측
        
        Args:
            df (pd.DataFrame): 예측할 데이터
            
        Returns:
            pd.DataFrame: 예측 결과
        """
        # 데이터가 충분한지 확인
        if len(df) <= self.input_window:
            logger.warning(f"예측을 위한 데이터가 부족합니다. 최소 {self.input_window+1}행 필요, 현재 {len(df)}행")
            return pd.DataFrame()
        
        # 데이터 정규화
        scaled_data = self.scaler.transform(df.values)
        
        # 시퀀스 데이터 생성
        X = []
        for i in range(len(scaled_data) - self.input_window):
            X.append(scaled_data[i:i + self.input_window])
        
        if not X:
            logger.warning("예측을 위한 시퀀스를 생성할 수 없습니다.")
            return pd.DataFrame()
        
        X = np.array(X)
        
        # 고장 확률 예측
        failure_probs = self.model.predict(X)
        
        # NaN 확인 및 처리
        if np.isnan(failure_probs).any():
            logger.warning("예측 결과에 NaN이 있습니다. 0으로 대체합니다.")
            failure_probs = np.nan_to_num(failure_probs, nan=0.0)
        
        # 결과 인덱스 확인
        if self.input_window >= len(df):
            logger.warning("결과 인덱스를 생성할 수 없습니다.")
            return pd.DataFrame()
        
        # 결과 데이터프레임 생성
        result_index = df.index[self.input_window:self.input_window + len(failure_probs)]
        
        # 예측 결과 반환
        results = pd.DataFrame({
            'timestamp': result_index,
            'failure_probability': failure_probs.flatten(),
            'is_failure_predicted': failure_probs.flatten() > 0.5
        })
        
        return results
    
    def plot_results(self, df, target_col, results, save_dir=None):
        """
        예측 결과 시각화 및 저장
        
        Args:
            df (pd.DataFrame): 원본 데이터
            target_col (str): 예측한 타겟 열
            results (pd.DataFrame): 예측 결과
            save_dir (str): 결과 저장 디렉토리
        """
        # 결과가 비어있는지 확인
        if results.empty:
            logger.warning("시각화를 위한 결과가 비어 있습니다.")
            return
        
        # 시각화를 위해 결과를 인덱싱
        results.set_index('timestamp', inplace=True)
        
        plt.figure(figsize=(15, 10))
        
        # 고장 확률 플롯
        plt.subplot(2, 1, 1)
        plt.plot(results.index, results['failure_probability'], label='고장 확률', color='blue')
        plt.axhline(y=0.5, color='r', linestyle='--', label='고장 임계값 (0.5)')
        
        # 고장 지점 강조
        failure_idx = results[results['is_failure_predicted'] == True].index
        if len(failure_idx) > 0:
            plt.scatter(failure_idx, results.loc[failure_idx, 'failure_probability'], color='red', label='예측된 고장')
            
        plt.title('고장 확률 추이')
        plt.ylabel('고장 확률')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 원본 데이터와 고장 지점
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df[target_col], label=f'원본 {target_col}', alpha=0.7)
        
        # 고장 지점 강조
        if len(failure_idx) > 0:
            for idx in failure_idx:
                if idx in df.index:
                    plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
            
        plt.title(f'{target_col} 원본 데이터 및 고장 예측')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 결과 저장
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, 'failure_prediction_results.png'), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def get_training_info(self):
        """
        학습 정보 반환
        
        Returns:
            dict: 학습 정보
        """
        if self.history is None:
            return {"status": "모델이 학습되지 않았습니다."}
        
        # 학습 정보 생성
        info = {
            "last_training_time": self.last_training_time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": len(self.history.history['loss']),
            "final_loss": float(self.history.history['loss'][-1]),
            "final_accuracy": float(self.history.history['accuracy'][-1]),
            "model_save_path": self.model_save_path
        }
        
        # 검증 데이터가 있는 경우에만 추가
        if 'val_loss' in self.history.history:
            info["final_val_loss"] = float(self.history.history['val_loss'][-1])
            info["final_val_accuracy"] = float(self.history.history['val_accuracy'][-1])
        
        return info

# =============================================================
# 2. LSTM 기반 자원 사용량 예측 모델
# =============================================================

class ResourceLSTM(BaseTimeSeriesModel):
    def __init__(self, input_dim, input_window=48, pred_horizon=24, lstm_units=64, dropout_rate=0.2, model_save_path='model_weights/resource_lstm.h5'):
        """
        LSTM 기반 자원 사용량 예측 모델
        
        Args:
            input_dim (int): 입력 특성 수
            input_window (int): 입력 시퀀스 길이
            pred_horizon (int): 예측 지평선
            lstm_units (int): LSTM 유닛 수
            dropout_rate (float): 드롭아웃 비율
            model_save_path (str): 모델 저장 경로
        """
        super().__init__(input_dim, input_window, pred_horizon, model_save_path)
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        self.prophet_model = None
        
        # 시나리오 정의
        self.scenarios = {
            '기본': 1.0,       # 기본 시나리오 (현재 추세)
            '저성장': 0.7,     # 저성장 시나리오
            '고성장': 1.5      # 고성장 시나리오
        }
    
    def _build_model(self):
        """LSTM 기반 자원 사용량 예측 모델 구축"""
        model = Sequential([
            # 입력층: (batch_size, time_steps, features)
            Bidirectional(LSTM(self.lstm_units, activation='tanh', return_sequences=True),
                          input_shape=(self.input_window, self.input_dim)),
            Dropout(self.dropout_rate),
            
            # 두 번째 LSTM 층
            LSTM(self.lstm_units, activation='tanh'),
            Dropout(self.dropout_rate),
            
            # 출력층 (예측 시퀀스)
            Dense(self.pred_horizon)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        return model
    
    def fit(self, df, target_col, epochs=100, batch_size=32, validation_split=0.2):
        """
        모델 학습
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            target_col (str): 예측할 타겟 열 이름
            epochs (int): 학습 에포크 수
            batch_size (int): 배치 크기
            validation_split (float): 검증 데이터 비율
            
        Returns:
            history: 학습 이력
        """
        # 학습 시작 시간 기록
        self.last_training_time = datetime.now()
        
        # 특성과 타겟 분리
        features = df.values
        
        # 타겟 열 인덱스 찾기
        target_col_idx = df.columns.get_loc(target_col)
        
        # 데이터 정규화
        scaled_features = self.scaler.fit_transform(features)
        
        # 데이터가 충분한지 확인
        required_samples = self.input_window + self.pred_horizon
        if len(scaled_features) < required_samples:
            logger.warning(f"데이터가 충분하지 않습니다. 최소 {required_samples}개 필요, 현재 {len(scaled_features)}개")
            self.history = None
            return None
        
        # 시퀀스 데이터 생성
        X, y = create_sequence_dataset(scaled_features, self.input_window, self.pred_horizon)
        
        # 타겟 열만 추출
        y_target = y[:, :, target_col_idx]
        
        # 학습/검증 데이터 분할
        if len(X) == 1:
            logger.warning("샘플이 1개뿐이라 검증을 건너뜁니다.")
            X_train, y_train = X, y_target
            X_val, y_val = None, None
            val_data = None
        else:
            train_size = max(int(len(X) * (1 - validation_split)), 1)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y_target[:train_size], y_target[train_size:]
            val_data = (X_val, y_val) if len(X_val) > 0 else None
        
        # 배치 크기 조정
        n_train = len(X_train)
        eff_bs = max(1, min(batch_size, n_train))
        if eff_bs != batch_size:
            logger.warning(f"batch_size({batch_size})를 train 샘플수({n_train})에 맞춰 {eff_bs}로 조정합니다.")
        
        # 콜백 설정
        callbacks = []
        if val_data is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # 모델 체크포인트
        if self.model_save_path:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            
            checkpoint = ModelCheckpoint(
                self.model_save_path,
                monitor='val_loss' if val_data else 'loss',
                save_best_only=True,
                mode='min'
            )
            callbacks.append(checkpoint)
        
        # 모델 학습
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=eff_bs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # 모델 저장
        if self.model_save_path:
            self.save_model()
        
        # Prophet 모델도 함께 학습 (장기 예측용)
        if len(df) >= 2:
            try:
                self._train_prophet_model(df, target_col)
            except Exception as e:
                logger.error(f"Prophet 모델 학습 실패: {e}")
                self.prophet_model = None
        else:
            logger.warning("Prophet 모델 학습을 위한 데이터가 부족합니다.")
            self.prophet_model = None
        
        return self.history
    
    def _train_prophet_model(self, df, target_col):
        """
        Prophet 모델 학습 (장기 예측용)
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            target_col (str): 예측할 타겟 열 이름
        """
        # Prophet 데이터 준비
        prophet_df = df.reset_index().rename(columns={df.index.name: 'ds', target_col: 'y'})
        prophet_df = prophet_df[['ds', 'y']]
        
        # 타임존 정보 제거 - Prophet 요구사항
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
        
        # Prophet 모델 생성 및 학습
        self.prophet_model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        
        # 추가 계절성 설정
        self.prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # 모델 학습
        self.prophet_model.fit(prophet_df)
        
    def predict(self, df, target_col):
        """
        자원 사용량 예측
        
        Args:
            df (pd.DataFrame): 예측할 데이터
            target_col (str): 예측할 타겟 열 이름
            
        Returns:
            pd.DataFrame: 예측 결과
        """
        # 데이터가 충분한지 확인
        if len(df) <= self.input_window:
            logger.warning(f"예측을 위한 데이터가 부족합니다. 최소 {self.input_window+1}행 필요, 현재 {len(df)}행")
            return pd.DataFrame()
        
        # 타겟 열 인덱스 찾기
        target_col_idx = df.columns.get_loc(target_col)
        
        # 데이터 정규화
        scaled_data = self.scaler.transform(df.values)
        
        # 시퀀스 데이터 생성
        X = []
        for i in range(len(scaled_data) - self.input_window):
            X.append(scaled_data[i:i + self.input_window])
        
        if not X:
            logger.warning("예측을 위한 시퀀스를 생성할 수 없습니다.")
            return pd.DataFrame()
        
        X = np.array(X)
        
        # 예측 수행
        predictions = self.model.predict(X)
        
        # 역정규화
        inverse_scaled = np.zeros((len(predictions), self.pred_horizon, df.shape[1]))
        for i, pred in enumerate(predictions):
            # 예측 값을 타겟 열에 할당
            temp = np.zeros((self.pred_horizon, df.shape[1]))
            temp[:, target_col_idx] = pred
            
            # 역정규화
            inverse_scaled[i] = self.scaler.inverse_transform(temp)
        
        # 결과 데이터프레임 생성
        result_index = df.index[self.input_window:self.input_window + len(predictions)]
        if len(result_index) == 0:
            logger.warning("결과 인덱스를 생성할 수 없습니다.")
            return pd.DataFrame()
        
        results = pd.DataFrame(index=result_index)
        
        # 예측 결과 추가
        results['predicted_value'] = inverse_scaled[:, 0, target_col_idx]  # 첫 번째 예측 시점 값
        
        # 마지막 시점의 예측 추가
        if len(result_index) > 0:
            last_date = result_index[-1]
            future_dates = [last_date + timedelta(hours=i+1) for i in range(self.pred_horizon)]
            
            future_df = pd.DataFrame({
                'date': future_dates,
                'predicted_value': inverse_scaled[-1, :, target_col_idx]
            })
            
            future_df.set_index('date', inplace=True)
            results = pd.concat([results, future_df])
        
        return results
    
    def predict_long_term(self, periods=180, capacity_threshold=None):
        """
        장기 자원 사용량 예측 (Prophet 사용)
        
        Args:
            periods (int): 예측 기간 (일)
            capacity_threshold (float): 용량 임계값
            
        Returns:
            dict: 시나리오별 예측 결과
            dict: 시나리오별 용량 증설 필요 시점
        """
        if self.prophet_model is None:
            logger.error("Prophet 모델이 학습되지 않았습니다.")
            # 빈 결과 반환
            return {}, {}
        
        try:
            # 미래 데이터프레임 생성
            future = self.prophet_model.make_future_dataframe(periods=periods, freq='D')
            
            # 기본 예측 수행
            forecast = self.prophet_model.predict(future)
            
            scenario_results = {}
            expansion_dates = {}
            
            # 각 시나리오별 시뮬레이션
            for scenario_name, growth_factor in self.scenarios.items():
                # 시나리오별 예측 조정
                scenario_forecast = forecast.copy()
                
                # 예측 시작점부터 성장률 조정
                forecast_start_date = scenario_forecast['ds'].iloc[-periods]
                future_mask = scenario_forecast['ds'] >= forecast_start_date
                
                # 트렌드 성분에 성장 계수 적용
                scenario_forecast.loc[future_mask, 'trend'] = scenario_forecast.loc[future_mask, 'trend'] * growth_factor
                
                # 최종 예측값 재계산
                for component in ['yearly', 'weekly', 'daily', 'monthly', 'additive_terms']:
                    if component in scenario_forecast.columns:
                        scenario_forecast.loc[future_mask, 'yhat'] = (
                            scenario_forecast.loc[future_mask, 'trend'] +
                            scenario_forecast.loc[future_mask, component]
                        )
                
                # 예측 구간 조정
                scenario_forecast.loc[future_mask, 'yhat_lower'] = scenario_forecast.loc[future_mask, 'yhat'] - \
                                                                (forecast.loc[future_mask, 'yhat_upper'] - 
                                                                 forecast.loc[future_mask, 'yhat'])
                
                scenario_forecast.loc[future_mask, 'yhat_upper'] = scenario_forecast.loc[future_mask, 'yhat'] + \
                                                                (forecast.loc[future_mask, 'yhat_upper'] - 
                                                                 forecast.loc[future_mask, 'yhat'])
                
                # 용량 초과 시점 찾기 (임계값이 제공된 경우)
                if capacity_threshold is not None:
                    over_capacity = scenario_forecast[scenario_forecast['yhat'] > capacity_threshold]
                    
                    if not over_capacity.empty:
                        expansion_date = over_capacity['ds'].iloc[0]
                        expansion_dates[scenario_name] = expansion_date
                        logger.info(f"시나리오 '{scenario_name}': 용량 증설 필요 시점: {expansion_date.strftime('%Y-%m-%d')}")
                    else:
                        expansion_dates[scenario_name] = None
                        logger.info(f"시나리오 '{scenario_name}': 예측 기간 내 용량 증설 필요 없음")
                
                scenario_results[scenario_name] = scenario_forecast
            
            return scenario_results, expansion_dates
            
        except Exception as e:
            logger.error(f"장기 예측 중 오류 발생: {e}")
            return {}, {}
        
    def get_training_info(self):
        """
        학습 정보 반환
        
        Returns:
            dict: 학습 정보
        """
        if self.history is None:
            return {"status": "모델이 학습되지 않았습니다."}
        
        # 학습 정보 생성
        info = {
            "last_training_time": self.last_training_time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": len(self.history.history['loss']),
            "final_loss": float(self.history.history['loss'][-1]),
            "final_mae": float(self.history.history['mae'][-1]),
            "model_save_path": self.model_save_path,
            "prophet_model": self.prophet_model is not None
        }
        
        # 검증 데이터가 있는 경우에만 추가
        if 'val_loss' in self.history.history:
            info["final_val_loss"] = float(self.history.history['val_loss'][-1])
            info["final_val_mae"] = float(self.history.history['val_mae'][-1])
        
        return info