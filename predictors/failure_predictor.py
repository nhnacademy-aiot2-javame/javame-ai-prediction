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
    """
    고장 예측 실행
    
    Args:
        config (dict): 설정 정보
        df (pd.DataFrame): 데이터프레임 (None이면 파일에서 로드)
        
    Returns:
        tuple: (성공 여부, 결과 데이터프레임)
    """
    logger.info("======= 고장 예측 실행 =======")
    
    failure_config = config["failure_prediction"]
    model_dir = config["results"].get("model_dir", "model_weights")
    ensure_dir(model_dir)
    
    # 데이터 확인
    if df is None or df.empty:
        logger.warning("데이터가 없습니다. 고장 예측을 건너뜁니다.")
        return False, None
    
    # 타겟 컬럼이 데이터에 있는지 확인 및 대체
    target_column = failure_config["target_column"]
    if target_column not in df.columns:
        logger.error(f"타겟 컬럼 '{target_column}'이 데이터에 없습니다.")
        logger.info(f"사용 가능한 컬럼: {df.columns.tolist()}")
        
        # 대체 컬럼 찾기
        target_candidates = [
            'temp_input',       # 온도
            'usage_user', 'user',  # CPU 사용자 사용률
            'usage_system', 'system',  # CPU 시스템 사용률
            'usage_iowait', 'iowait',  # CPU I/O 대기
            'used_percent'      # 사용률(메모리, 디스크)
        ]
        
        for candidate in target_candidates:
            if candidate in df.columns:
                logger.info(f"'{candidate}' 컬럼을 타겟으로 사용합니다.")
                failure_config["target_column"] = candidate
                break
        else:
            logger.error("적절한 타겟 컬럼을 찾을 수 없습니다.")
            return False, None
    
    # 업데이트된 타겟 컬럼 사용
    target_column = failure_config["target_column"]
    
    # 데이터 전처리
    df = preprocess_data(df, target_column)
    
    # NaN 값 확인 및 처리
    if df.isna().any().any():
        logger.warning("데이터에 NaN 값이 있습니다. 보간합니다.")
        df = df.interpolate(method='linear')
        df = df.bfill()
        df = df.ffill()
    
    # 설정이 데이터 크기에 적합한지 확인 및 조정
    config = adjust_config_for_data_size(config, df)
    failure_config = config["failure_prediction"]
    
    # 충분한 데이터가 있는지 확인
    min_required_samples = 2  # 최소한의 학습 가능 데이터 수
    if len(df) < min_required_samples:
        logger.error(f"데이터가 너무 적습니다: {len(df)}행. 최소 {min_required_samples}행 필요합니다.")
        return False, None
    
    # 데이터 크기에 맞게 input_window 조정
    input_window = min(failure_config["input_window"], len(df) - 1)
    
    # 고급 특성 생성 (추가된 부분)
    df = _create_advanced_features(df)
    
    # 모델 생성
    input_dim = df.shape[1]
    model = FailureLSTM(
        input_dim=input_dim,
        input_window=input_window,
        pred_horizon=1,
        model_save_path=os.path.join(model_dir, 'failure_lstm.h5')
    )
    
    # 모델 학습
    logger.info("고장 예측 모델 학습 중...")
    
    # 임계값 조정 (데이터에 맞게)
    target_values = df[target_column].values
    mean_value = target_values.mean()
    std_value = target_values.std()
    threshold = mean_value + 1.5 * std_value  # 평균 + 1.5 * 표준편차
    
    history = model.fit(
        df, 
        target_col=target_column,
        failure_threshold=threshold,
        epochs=50,
        batch_size=16  # 작은 데이터셋에 맞게 배치 크기 조정
    )
    
    # 학습이 실패한 경우
    if history is None:
        logger.error("모델 학습에 실패했습니다. 데이터가 충분하지 않습니다.")
        return False, None
    
    # 데이터가 충분하지 않으면 예측 단계 건너뛰기
    if len(df) <= input_window:
        logger.warning(f"예측을 위한 데이터가 부족합니다. 최소 {input_window+1}행 필요, 현재 {len(df)}행")
        return False, None
    
    # 예측 수행
    logger.info("고장 예측 수행 중...")
    try:
        results = model.predict(df)
        
        # 결과가 비어있는지 확인
        if results.empty:
            logger.warning("예측 결과가 비어 있습니다.")
            return False, None
        
        # 디바이스 ID 추가
        device_id = "device_001"  # 기본값
        results['device_id'] = device_id
        
        # 예측 시간 추가
        prediction_time = datetime.now()
        results['prediction_time'] = prediction_time
        
        # 결과 저장 (MySQL)
        if 'mysql' in config:
            try:
                save_to_mysql(results, "failure_predictions", config)
            except Exception as e:
                logger.error(f"MySQL 저장 중 오류: {e}")
                # MySQL 오류가 발생해도 계속 진행
        
        # 결과 시각화 및 저장
        _save_prediction_results(config, df, target_column, results, prediction_time, model)
        
        logger.info(f"고장 예측 완료: 총 {len(results)}개 결과, 예측된 고장: {results['is_failure_predicted'].sum()}개")
        return True, results
    
    except Exception as e:
        logger.error(f"예측 수행 중 오류 발생: {e}")
        return False, None
def run_failure_prediction(config, df=None):
    """
    고장 예측 실행
    
    Args:
        config (dict): 설정 정보
        df (pd.DataFrame): 데이터프레임 (None이면 파일에서 로드)
        
    Returns:
        tuple: (성공 여부, 결과 데이터프레임)
    """
    logger.info("======= 고장 예측 실행 =======")
    
    failure_config = config["failure_prediction"]
    model_dir = config["results"].get("model_dir", "model_weights")
    ensure_dir(model_dir)
    
    # 데이터 확인
    if df is None or df.empty:
        logger.warning("데이터가 없습니다. 고장 예측을 건너뜁니다.")
        return False, None
    
    # 타겟 컬럼이 데이터에 있는지 확인 및 대체
    target_column = failure_config["target_column"]
    if target_column not in df.columns:
        logger.error(f"타겟 컬럼 '{target_column}'이 데이터에 없습니다.")
        logger.info(f"사용 가능한 컬럼: {df.columns.tolist()}")
        
        # 대체 컬럼 찾기
        target_candidates = [
            'temp_input',       # 온도
            'usage_user', 'user',  # CPU 사용자 사용률
            'usage_system', 'system',  # CPU 시스템 사용률
            'usage_iowait', 'iowait',  # CPU I/O 대기
            'used_percent'      # 사용률(메모리, 디스크)
        ]
        
        for candidate in target_candidates:
            if candidate in df.columns:
                logger.info(f"'{candidate}' 컬럼을 타겟으로 사용합니다.")
                failure_config["target_column"] = candidate
                break
        else:
            logger.error("적절한 타겟 컬럼을 찾을 수 없습니다.")
            return False, None
    
    # 업데이트된 타겟 컬럼 사용
    target_column = failure_config["target_column"]
    
    # 데이터 전처리
    df = preprocess_data(df, target_column)
    
    # NaN 값 확인 및 처리
    if df.isna().any().any():
        logger.warning("데이터에 NaN 값이 있습니다. 보간합니다.")
        df = df.interpolate(method='linear')
        df = df.bfill()
        df = df.ffill()
    
    # 설정이 데이터 크기에 적합한지 확인 및 조정
    config = adjust_config_for_data_size(config, df)
    failure_config = config["failure_prediction"]
    
    # 충분한 데이터가 있는지 확인
    min_required_samples = 2  # 최소한의 학습 가능 데이터 수
    if len(df) < min_required_samples:
        logger.error(f"데이터가 너무 적습니다: {len(df)}행. 최소 {min_required_samples}행 필요합니다.")
        return False, None
    
    # 고급 특성 생성 (추가된 부분)
    df = _create_advanced_features(df)
    
    # 데이터 크기에 맞게 input_window 조정
    input_window = min(failure_config["input_window"], len(df) - 1)
    
    # 모델 생성
    input_dim = df.shape[1]
    model = FailureLSTM(
        input_dim=input_dim,
        input_window=input_window,
        pred_horizon=1,
        model_save_path=os.path.join(model_dir, 'failure_lstm.h5')
    )
    
    # 모델 학습
    logger.info("고장 예측 모델 학습 중...")
    
    # 임계값 조정 (데이터에 맞게)
    target_values = df[target_column].values
    mean_value = target_values.mean()
    std_value = target_values.std()
    threshold = mean_value + 1.5 * std_value  # 평균 + 1.5 * 표준편차
    
    history = model.fit(
        df, 
        target_col=target_column,
        failure_threshold=threshold,
        epochs=50,
        batch_size=16  # 작은 데이터셋에 맞게 배치 크기 조정
    )
    
    # 학습이 실패한 경우
    if history is None:
        logger.error("모델 학습에 실패했습니다. 데이터가 충분하지 않습니다.")
        return False, None
    
    # 데이터가 충분하지 않으면 예측 단계 건너뛰기
    if len(df) <= input_window:
        logger.warning(f"예측을 위한 데이터가 부족합니다. 최소 {input_window+1}행 필요, 현재 {len(df)}행")
        return False, None
    
    # 예측 수행
    logger.info("고장 예측 수행 중...")
    try:
        results = model.predict(df)
        
        # 결과가 비어있는지 확인
        if results.empty:
            logger.warning("예측 결과가 비어 있습니다.")
            return False, None
        
        # 디바이스 ID 추가
        device_id = "device_001"  # 기본값
        results['device_id'] = device_id
        
        # 예측 시간 추가
        prediction_time = datetime.now()
        results['prediction_time'] = prediction_time
        
        # 결과 저장 (MySQL)
        if 'mysql' in config:
            try:
                save_to_mysql(results, "failure_predictions", config)
            except Exception as e:
                logger.error(f"MySQL 저장 중 오류: {e}")
                # MySQL 오류가 발생해도 계속 진행
        
        # 결과 시각화 및 저장
        _save_prediction_results(config, df, target_column, results, prediction_time, model)
        
        # API로 결과 전송
        try:
            api_sender = APISender(get_api_url())
            api_success = api_sender.send_failure_prediction(results)
            if api_success:
                logger.info("API로 고장 예측 결과 전송 성공")
            else:
                logger.warning("API로 고장 예측 결과 전송 실패")
        except Exception as e:
            logger.error(f"API 결과 전송 중 오류: {e}")
            # API 오류가 발생해도 계속 진행
        
        logger.info(f"고장 예측 완료: 총 {len(results)}개 결과, 예측된 고장: {results['is_failure_predicted'].sum()}개")
        return True, results
    
    except Exception as e:
        logger.error(f"예측 수행 중 오류 발생: {e}")
        return False, None        
def _create_advanced_features(df):
    """
    고급 특성 생성
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        
    Returns:
        pd.DataFrame: 고급 특성이 추가된 데이터프레임
    """
    # 기존 데이터프레임 복사
    df_advanced = df.copy()
    
    try:
        # 1. CPU 관련 복합 지표
        if 'usage_user' in df.columns and 'usage_system' in df.columns:
            # CPU 총 사용률: 사용자 + 시스템
            df_advanced['cpu_total_usage'] = df['usage_user'] + df['usage_system']
        
        if 'usage_idle' in df.columns:
            # CPU 부하율: 100 - 유휴율
            df_advanced['cpu_load'] = 100 - df['usage_idle']
        
        # 2. I/O 대기와 디스크 활동 관계
        if 'usage_iowait' in df.columns:
            # I/O 대기 비율에 대한 변화율
            df_advanced['iowait_change'] = df['usage_iowait'].diff().fillna(0)
            
            # I/O 대기 이동 평균 (추세 파악)
            df_advanced['iowait_ma'] = df['usage_iowait'].rolling(window=5, min_periods=1).mean()
        
        # 3. 메모리 관련 지표
        if 'used_percent' in df.columns and 'available_percent' in df.columns:
            # 메모리 스트레스 지수: 사용률 / 가용률
            df_advanced['memory_stress'] = df['used_percent'] / df['available_percent'].clip(lower=0.1)
        
        # 4. 온도 관련 지표
        if 'temp_input' in df.columns:
            # 온도 변화율
            df_advanced['temp_change'] = df['temp_input'].diff().fillna(0)
            
            # 온도 이동 평균
            df_advanced['temp_ma'] = df['temp_input'].rolling(window=5, min_periods=1).mean()
            
            # 온도 급상승 지표 (이전 5포인트보다 높은지)
            df_advanced['temp_surge'] = (
                df['temp_input'] > df['temp_input'].shift(1).rolling(window=5, min_periods=1).max()
            ).astype(float)
        
        # 5. 시간 기반 특성
        # 타임스탬프에서 시간 정보 추출
        if isinstance(df.index, pd.DatetimeIndex):
            df_advanced['hour'] = df.index.hour
            df_advanced['day_of_week'] = df.index.dayofweek
            
            # 피크 시간대 플래그 (주간 9-17시)
            df_advanced['is_peak_hour'] = ((df.index.hour >= 9) & (df.index.hour <= 17)).astype(float)
        
        # 결측치 처리
        df_advanced = df_advanced.fillna(0)
        
        logger.info(f"고급 특성 생성 완료: {len(df.columns)}개 → {len(df_advanced.columns)}개 특성")
        return df_advanced
        
    except Exception as e:
        logger.error(f"고급 특성 생성 중 오류 발생: {e}")
        # 오류 발생 시 원본 데이터 반환
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