#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
고장 예측(Failure Prediction) 모듈

run_prediction.py에서 분리된 고장 예측 관련 기능.
센서 데이터를 분석하여 설비 고장을 예측합니다.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from utils.api_sender import APISender, get_api_url

# 모델 임포트
from models.lstm_model import FailureLSTM

# 유틸리티 함수 임포트
from utils.visualization import plot_sensor_data, plot_failure_prediction, plot_correlation_matrix
from utils.data_loader import preprocess_data, adjust_config_for_data_size
from utils.db_utils import save_to_mysql

# 로깅 설정
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"디렉토리 생성: {directory}")

def run_failure_prediction(config, df=None):
    """고장 예측 실행"""
    logger.info("======= 고장 예측 실행 =======")
    
    failure_config = config["failure_prediction"]
    model_dir = config["results"].get("model_dir", "model_weights")
    ensure_dir(model_dir)
    
    # 데이터 확인
    if df is None or df.empty:
        logger.warning("데이터가 없습니다. 고장 예측을 건너뜁니다.")
        return False, None
    
    # 온도 기반 고장 예측을 위한 타겟 후보
    target_candidates = [
        'sensor_data_temperature',  # 외부 센서 온도
        'temp_input',               # 서버 내부 온도
        'high_temp_high_cpu',       # 고온-고부하 복합 특성
        'high_temp_low_mem'         # 고온-메모리부족 복합 특성
    ]
    
    # 타겟 컬럼 선택
    target_column = failure_config["target_column"]
    if target_column not in df.columns:
        logger.error(f"타겟 컬럼 '{target_column}'이 데이터에 없습니다.")
        
        # 대체 타겟 찾기
        for candidate in target_candidates:
            if candidate in df.columns:
                target_column = candidate
                logger.info(f"타겟 컬럼을 '{target_column}'으로 변경합니다.")
                failure_config["target_column"] = target_column
                break
        else:
            logger.error("적절한 타겟 컬럼을 찾을 수 없습니다.")
            return False, None
    
    # 데이터 전처리
    df = preprocess_data(df, target_column)
    
    # 고급 특성 생성
    df = _create_advanced_features(df)
    
    # 입력 윈도우 설정
    input_window = min(failure_config["input_window"], len(df) // 2)
    
    # 모델 생성
    input_dim = df.shape[1]
    model = FailureLSTM(
        input_dim=input_dim,
        input_window=input_window,
        pred_horizon=1,
        model_save_path=os.path.join(model_dir, 'failure_lstm.h5')
    )
    
    # 임계값 설정 (온도 기반)
    if 'temperature' in target_column:
        # 온도는 높을수록 위험
        values = df[target_column].values
        threshold = np.percentile(values, 90)  # 상위 10% 온도를 위험으로 간주
    else:
        # 기본 임계값 사용
        threshold = failure_config["threshold"]
    
    # 모델 학습
    logger.info(f"고장 예측 모델 학습 중 (타겟: {target_column}, 임계값: {threshold})...")
    history = model.fit(
        df, 
        target_col=target_column,
        failure_threshold=threshold,
        epochs=50,
        batch_size=16
    )
    
    # 예측 수행
    logger.info("고장 예측 수행 중...")
    results = model.predict(df)
    
    # 결과 저장
    if not results.empty:
        # 디바이스 ID 및 예측 시간 추가
        results['device_id'] = "server_001"  # 적절한 서버 ID로 변경
        results['prediction_time'] = datetime.now()
        
        # MySQL 저장
        if 'mysql' in config:
            save_to_mysql(results, "failure_predictions", config)
        
        # 결과 시각화 및 저장
        _save_failure_prediction_results(
            config, df, target_column, results, 
            datetime.now(), model
        )
        
        # 고장 예측 횟수 계산
        failure_count = results['is_failure_predicted'].sum()
        logger.info(f"고장 예측 완료: {len(results)}개 데이터 포인트 중 {failure_count}개 고장 예측")
        
        return True, results
    else:
        logger.warning("예측 결과가 비어 있습니다.")
        return False, None
def _create_advanced_features(df):
    """고급 특성 생성"""
    # 기존 데이터프레임 복사
    df_advanced = df.copy()
    
    try:
        # 1. 서버 리소스 관련 특성
        # CPU 관련 특성
        if 'usage_user' in df.columns:
            # CPU 사용률 변화율
            df_advanced['usage_user_change'] = df['usage_user'].diff().fillna(0)
            # CPU 사용률 이동 평균
            df_advanced['usage_user_ma'] = df['usage_user'].rolling(window=6, min_periods=1).mean()
            # CPU 사용률 이동 표준편차 (변동성)
            df_advanced['usage_user_std'] = df['usage_user'].rolling(window=6, min_periods=1).std().fillna(0)
            # CPU 고부하 상태
            df_advanced['high_cpu'] = (df['usage_user'] > 80).astype(float)
        
        # 메모리 관련 특성
        if 'available_percent' in df.columns:
            # 메모리 가용률 변화율
            df_advanced['available_percent_change'] = df['available_percent'].diff().fillna(0)
            # 메모리 가용률 이동 평균
            df_advanced['available_percent_ma'] = df['available_percent'].rolling(window=6, min_periods=1).mean()
            # 낮은 메모리 가용률 상태
            df_advanced['low_memory'] = (df['available_percent'] < 20).astype(float)
            # 메모리 사용률 (가용률의 반대)
            df_advanced['memory_usage'] = 100 - df['available_percent']
        
        # 시스템 로드 관련 특성
        if 'load1' in df.columns:
            # 로드 변화율
            df_advanced['load1_change'] = df['load1'].diff().fillna(0)
            # 로드 이동 평균
            df_advanced['load1_ma'] = df['load1'].rolling(window=6, min_periods=1).mean()
            # 코어 수 대비 고부하 (4코어 가정)
            cpu_cores = 4  # 서버 코어 수에 맞게 조정
            df_advanced['high_load'] = (df['load1'] > cpu_cores * 0.7).astype(float)
        
        # 2. 환경 데이터 관련 특성
        # 온도 관련 특성
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]
        for col in temp_cols:
            # 온도 변화율
            df_advanced[f'{col}_change'] = df[col].diff().fillna(0)
            # 온도 이동 평균
            df_advanced[f'{col}_ma'] = df[col].rolling(window=6, min_periods=1).mean()
            # 고온 상태
            df_advanced[f'high_{col}'] = (df[col] > 28).astype(float)  # 28도 이상을 고온으로 가정
        
        # 습도 관련 특성
        if 'sensor_data_humidity' in df.columns:
            # 습도 변화율
            df_advanced['humidity_change'] = df['sensor_data_humidity'].diff().fillna(0)
            # 습도 이동 평균
            df_advanced['humidity_ma'] = df['sensor_data_humidity'].rolling(window=6, min_periods=1).mean()
            # 고습 상태
            df_advanced['high_humidity'] = (df['sensor_data_humidity'] > 70).astype(float)  # 70% 이상을 고습으로 가정
        
        # 3. 서버-환경 상관관계 특성
        # 온도와 CPU 사용률 관계
        if 'sensor_data_temperature' in df.columns and 'usage_user' in df.columns:
            # 온도 대비 CPU 사용률 비율
            df_advanced['temp_cpu_ratio'] = df['sensor_data_temperature'] / df['usage_user'].clip(lower=1)
            # 온도가 높을 때 CPU 사용률이 높은 상태
            df_advanced['high_temp_high_cpu'] = ((df['sensor_data_temperature'] > 28) & 
                                              (df['usage_user'] > 70)).astype(float)
        
        # 온도와 메모리 가용률 관계
        if 'sensor_data_temperature' in df.columns and 'available_percent' in df.columns:
            # 온도가 높을 때 메모리 가용률이 낮은 상태
            df_advanced['high_temp_low_mem'] = ((df['sensor_data_temperature'] > 28) & 
                                             (df['available_percent'] < 30)).astype(float)
        
        # 4. 시간 기반 특성
        if isinstance(df.index, pd.DatetimeIndex):
            # 시간대 특성
            df_advanced['hour'] = df.index.hour
            df_advanced['dayofweek'] = df.index.dayofweek
            df_advanced['month'] = df.index.month
            
            # 주기성 표현
            df_advanced['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df_advanced['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df_advanced['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df_advanced['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            
            # 업무 시간대
            df_advanced['is_business_hour'] = ((df.index.hour >= 9) & 
                                            (df.index.hour < 18) & 
                                            (df.index.dayofweek < 5)).astype(float)
        
        # 결측치 제거
        df_advanced = df_advanced.fillna(0)
        
        logger.info(f"고급 특성 생성 완료: {len(df.columns)}개 → {len(df_advanced.columns)}개 특성")
        return df_advanced
        
    except Exception as e:
        logger.error(f"고급 특성 생성 중 오류 발생: {e}")
        return df

def _save_prediction_results(config, df, target_column, results, prediction_time, model):
    """
    예측 결과 저장 및 시각화
    
    Args:
        config (dict): 설정 정보
        df (pd.DataFrame): 원본 데이터
        target_column (str): 타겟 열
        results (pd.DataFrame): 예측 결과
        prediction_time (datetime): 예측 시간
        model (FailureLSTM): 학습된 모델
    """
    if config["results"]["save_plots"]:
        save_dir = os.path.join(
            config["results"]["save_dir"],
            "failure_prediction",
            prediction_time.strftime("%Y%m%d_%H%M%S")
        )
        ensure_dir(save_dir)
        
        # 원본 데이터 시각화
        plot_sensor_data(
            df, 
            cols=[target_column], 
            save_path=os.path.join(save_dir, "sensor_data.png")
        )
        
        # 상관관계 행렬 시각화
        plot_correlation_matrix(
            df,
            save_path=os.path.join(save_dir, "correlation_matrix.png")
        )
        
        # 예측 결과 시각화
        plot_failure_prediction(
            df, 
            target_column, 
            results, 
            threshold=config["failure_prediction"]["threshold"],
            save_path=os.path.join(save_dir, "failure_prediction_results.png")
        )
        
        # 학습 정보 저장
        training_info = model.get_training_info()
        with open(os.path.join(save_dir, "training_info.json"), 'w') as f:
            json.dump(training_info, f, indent=4)
    
    # CSV 파일로 저장
    if config["results"]["save_csv"]:
        csv_dir = os.path.join(
            config["results"]["save_dir"],
            "failure_prediction",
            "csv"
        )
        ensure_dir(csv_dir)
        
        csv_path = os.path.join(
            csv_dir,
            f"failure_predictions_{prediction_time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        results.to_csv(csv_path)
        logger.info(f"예측 결과 CSV 저장: {csv_path}")