#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
시계열 예측 모델을 위한 기본 인터페이스

모든 예측 모델 클래스가 구현해야 하는 공통 인터페이스 정의
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseTimeSeriesModel:
    """시계열 예측 모델을 위한 기본 인터페이스"""
    
    def __init__(self, input_dim, input_window, pred_horizon=1):
        """
        기본 모델 초기화
        
        Args:
            input_dim (int): 입력 특성 수
            input_window (int): 입력 시퀀스 길이
            pred_horizon (int): 예측 지평선
        """
        self.input_dim = input_dim
        self.input_window = input_window
        self.pred_horizon = pred_horizon
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        self.last_training_time = None
    
    def _build_model(self):
        """
        모델 아키텍처 구축
        
        Returns:
            tf.keras.Model: 구축된 모델
        
        Raises:
            NotImplementedError: 서브클래스에서 구현해야 함
        """
        raise NotImplementedError("서브클래스에서 구현해야 합니다")
    
    def fit(self, df, target_col, **kwargs):
        """
        모델 학습
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            target_col (str): 예측할 타겟 열 이름
            **kwargs: 추가 학습 매개변수
            
        Returns:
            tf.keras.callbacks.History: 학습 이력
        
        Raises:
            NotImplementedError: 서브클래스에서 구현해야 함
        """
        raise NotImplementedError("서브클래스에서 구현해야 합니다")
    
    def predict(self, df, target_col=None):
        """
        예측 수행
        
        Args:
            df (pd.DataFrame): 예측할 데이터
            target_col (str, optional): 예측할 타겟 열 이름
            
        Returns:
            pd.DataFrame: 예측 결과
        
        Raises:
            NotImplementedError: 서브클래스에서 구현해야 함
        """
        raise NotImplementedError("서브클래스에서 구현해야 합니다")
    
    def plot_results(self, df, target_col, results, save_dir=None):
        """
        결과 시각화
        
        Args:
            df (pd.DataFrame): 원본 데이터
            target_col (str): 예측한 타겟 열
            results (pd.DataFrame): 예측 결과
            save_dir (str, optional): 결과 저장 디렉토리
        
        Raises:
            NotImplementedError: 서브클래스에서 구현해야 함
        """
        raise NotImplementedError("서브클래스에서 구현해야 합니다")
    
    def get_training_info(self):
        """
        학습 정보 반환
        
        Returns:
            dict: 학습 정보
        
        Raises:
            NotImplementedError: 서브클래스에서 구현해야 함
        """
        raise NotImplementedError("서브클래스에서 구현해야 합니다")

    def save_model(self, path):
        """
        모델 저장
        
        Args:
            path (str): 저장 경로
            
        Returns:
            bool: 성공 여부
        """
        if self.model is not None:
            try:
                self.model.save(path)
                logger.info(f"모델 저장 완료: {path}")
                return True
            except Exception as e:
                logger.error(f"모델 저장 중 오류 발생: {e}")
        
        logger.warning("저장할 모델이 없습니다.")
        return False
    
    def load_model(self, path):
        """
        모델 로드
        
        Args:
            path (str): 로드할 모델 경로
            
        Returns:
            bool: 성공 여부
        """
        if os.path.exists(path):
            try:
                self.model = tf.keras.models.load_model(path)
                logger.info(f"모델 로드 완료: {path}")
                return True
            except Exception as e:
                logger.error(f"모델 로드 중 오류 발생: {e}")
        else:
            logger.error(f"모델 파일이 존재하지 않습니다: {path}")
        
        return False

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