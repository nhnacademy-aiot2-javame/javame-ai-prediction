#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
자원 사용량 분석(Resource Usage Analysis) 모듈

run_prediction.py에서 분리된 자원 사용량 분석 관련 기능.
자원 사용량을 예측하고 용량 증설 계획을 수립합니다.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# 모델 임포트
from models.lstm_model import ResourceLSTM

# 유틸리티 함수 임포트
from utils.visualization import plot_sensor_data, plot_resource_prediction, plot_correlation_matrix, plot_long_term_scenarios
from utils.data_loader import preprocess_data, adjust_config_for_data_size
from utils.db_utils import save_to_mysql

# 로깅 설정
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"디렉토리 생성: {directory}")

def run_resource_analysis(config, df=None):
    """자원 사용량 분석 실행"""
    logger.info("======= 자원 사용량 분석 실행 =======")
    
    resource_config = config["resource_analysis"]
    model_dir = config["results"].get("model_dir", "model_weights")
    ensure_dir(model_dir)
    
    # 데이터 확인
    if df is None or df.empty:
        logger.warning("데이터가 없습니다. 자원 사용량 분석을 건너뜁니다.")
        return False, None
    
    # 타겟 컬럼 후보 목록 (새 데이터 구조 반영)
    target_candidates = [
        'usage_user',           # CPU 사용률
        'available_percent',    # 메모리 가용률
        'memory_usage',         # 메모리 사용률 (계산된 특성)
        'load1'                 # 시스템 로드
    ]
    
    # 타겟 컬럼 선택
    target_column = resource_config["target_column"]
    if target_column not in df.columns:
        logger.error(f"타겟 컬럼 '{target_column}'이 데이터에 없습니다.")
        
        # 대체 타겟 찾기
        for candidate in target_candidates:
            if candidate in df.columns:
                target_column = candidate
                logger.info(f"타겟 컬럼을 '{target_column}'으로 변경합니다.")
                resource_config["target_column"] = target_column
                break
        else:
            logger.error("적절한 타겟 컬럼을 찾을 수 없습니다.")
            return False, None
    
    # 데이터 전처리
    df = preprocess_data(df, target_column)
    
    # 고급 특성 생성
    df = _create_advanced_features(df)
    
    # 입력 윈도우 및 예측 지평선 설정
    input_window = min(resource_config["input_window"], len(df) // 2)
    pred_horizon = 24  # 24시간 예측
    
    # 모델 생성
    input_dim = df.shape[1]
    model = ResourceLSTM(
        input_dim=input_dim,
        input_window=input_window,
        pred_horizon=pred_horizon,
        model_save_path=os.path.join(model_dir, 'resource_lstm.h5')
    )
    
    # 모델 학습
    logger.info(f"{target_column} 예측 모델 학습 중...")
    history = model.fit(
        df, 
        target_col=target_column,
        epochs=50,
        batch_size=16
    )
    
    # 예측 수행
    logger.info(f"{target_column} 예측 수행 중...")
    results = model.predict(df, target_column)
    
    # 결과 저장
    if not results.empty:
        # 결과에 메타데이터 추가
        results_df = results.reset_index()
        results_df.rename(columns={'index': 'timestamp'}, inplace=True)
        results_df['device_id'] = "server_001"  # 적절한 서버 ID로 변경
        results_df['resource_type'] = target_column
        results_df['prediction_time'] = datetime.now()
        
        # MySQL 저장
        if 'mysql' in config:
            save_to_mysql(results_df, "resource_predictions", config)
        
        # 장기 예측 수행
        logger.info("장기 자원 사용량 예측 수행 중...")
        scenario_results, expansion_dates = model.predict_long_term(
            periods=90,  # 3개월
            capacity_threshold=resource_config["capacity_threshold"]
        )
        
        # 예측 결과 시각화 및 저장
        _save_prediction_results(
            config, df, target_column, results, 
            scenario_results, expansion_dates, 
            datetime.now(), model
        )
        
        logger.info(f"{target_column} 예측 완료")
        return True, {
            'short_term': results,
            'long_term': scenario_results,
            'expansion_dates': expansion_dates
        }
    else:
        logger.warning("예측 결과가 비어 있습니다.")
        return False, None

def _create_advanced_features(df):
    """
    자원 분석을 위한 고급 특성 생성
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        
    Returns:
        pd.DataFrame: 고급 특성이 추가된 데이터프레임
    """
    # 기존 데이터프레임 복사
    df_advanced = df.copy()
    
    try:
        # 1. CPU 관련 복합 지표
        if all(col in df.columns for col in ['usage_user', 'usage_system']):
            # CPU 총 사용률: 사용자 + 시스템
            df_advanced['cpu_total_usage'] = df['usage_user'] + df['usage_system']
        
        if 'usage_idle' in df.columns:
            # CPU 부하율: 100 - 유휴율
            df_advanced['cpu_load'] = 100 - df['usage_idle']
            
            # CPU 부하 상태 (높음/중간/낮음)
            df_advanced['cpu_high_load'] = (df_advanced['cpu_load'] > 80).astype(float)
            df_advanced['cpu_medium_load'] = ((df_advanced['cpu_load'] > 50) & 
                                            (df_advanced['cpu_load'] <= 80)).astype(float)
        
        # 2. I/O 대기와 디스크 활동 관계
        if 'usage_iowait' in df.columns:
            # I/O 대기 비율에 대한 변화율
            df_advanced['iowait_change'] = df['usage_iowait'].diff().fillna(0)
            
            # I/O 대기 이동 평균 (추세 파악)
            df_advanced['iowait_ma'] = df['usage_iowait'].rolling(window=5, min_periods=1).mean()
            
            # I/O 대기 급증 지표 (전 시간의 2배 이상)
            df_advanced['iowait_surge'] = ((df['usage_iowait'] > 5) & 
                                         (df['usage_iowait'] > 2 * df['usage_iowait'].shift(1))).astype(float)
        
        # 3. 디스크 용량 관련 지표
        if 'disk_usedPercent_used_percent' in df.columns:
            # 디스크 사용률 변화 속도 (시간당)
            df_advanced['disk_usage_change'] = df['disk_usedPercent_used_percent'].diff().fillna(0)
            
            # 디스크 임계치 접근도 (90% 기준)
            df_advanced['disk_threshold_proximity'] = (90 - df['disk_usedPercent_used_percent']).clip(lower=0) / 90
            
            # 디스크 고갈 위험도 (80% 이상이면 위험)
            df_advanced['disk_high_usage'] = (df['disk_usedPercent_used_percent'] > 80).astype(float)
        
        # 4. 메모리 관련 지표
        if 'mem_usedPercent_used_percent' in df.columns:
            # 메모리 사용률 변화 속도
            df_advanced['memory_usage_change'] = df['mem_usedPercent_used_percent'].diff().fillna(0)
            
            # 메모리 임계치 접근도 (90% 기준)
            df_advanced['memory_threshold_proximity'] = (90 - df['mem_usedPercent_used_percent']).clip(lower=0) / 90
            
            # 메모리 고갈 위험도 (85% 이상이면 위험)
            df_advanced['memory_high_usage'] = (df['mem_usedPercent_used_percent'] > 85).astype(float)
        
        # 5. 시스템 로드 관련 지표
        if 'load1' in df.columns:
            # 부하 변화율 (급등 감지)
            df_advanced['load_change'] = df['load1'].diff().fillna(0)
            
            # 부하 이동 평균 (추세 파악)
            df_advanced['load_ma'] = df['load1'].rolling(window=5, min_periods=1).mean()
            
            # 고부하 상태 (CPU 코어 수에 따라 설정 필요)
            # 기본 값은 4코어 시스템 가정 (load1 > 3.5)
            cpu_cores = 4  # 실제 환경에 맞게 조정 필요
            df_advanced['high_load'] = (df['load1'] > cpu_cores * 0.8).astype(float)
        
        # 6. 시간 기반 특성
        # 타임스탬프에서 시간 정보 추출
        if isinstance(df.index, pd.DatetimeIndex):
            df_advanced['hour'] = df.index.hour
            df_advanced['day_of_week'] = df.index.dayofweek
            
            # 피크 시간대 플래그 (주간 9-17시)
            df_advanced['is_peak_hour'] = ((df.index.hour >= 9) & (df.index.hour <= 17)).astype(float)
            
            # 주중/주말 구분
            df_advanced['is_weekday'] = (df.index.dayofweek < 5).astype(float)
        
        # 7. 자원 균형 지표 (CPU vs 메모리 vs 디스크)
        if all(col in df_advanced.columns for col in ['cpu_load', 'mem_usedPercent_used_percent']):
            # CPU-메모리 불균형 지표 (차이가 크면 불균형)
            df_advanced['cpu_memory_imbalance'] = abs(df_advanced['cpu_load'] - 
                                                   df_advanced.get('mem_usedPercent_used_percent', 0))
        
        # 결측치 처리
        df_advanced = df_advanced.fillna(0)
        
        logger.info(f"자원 분석용 고급 특성 생성 완료: {len(df.columns)}개 → {len(df_advanced.columns)}개 특성")
        return df_advanced
        
    except Exception as e:
        logger.error(f"고급 특성 생성 중 오류 발생: {e}")
        # 오류 발생 시 원본 데이터 반환
        return df

def _save_prediction_results(config, df, target_column, results, scenario_results, 
                           expansion_dates, prediction_time, model):
    """
    예측 결과 저장 및 시각화
    
    Args:
        config (dict): 설정 정보
        df (pd.DataFrame): 원본 데이터
        target_column (str): 타겟 열
        results (pd.DataFrame): 예측 결과
        scenario_results (dict): 시나리오별 장기 예측 결과
        expansion_dates (dict): 시나리오별 용량 증설 필요 시점
        prediction_time (datetime): 예측 시간
        model (ResourceLSTM): 학습된 모델
    """
    if config["results"]["save_plots"]:
        save_dir = os.path.join(
            config["results"]["save_dir"],
            "resource_analysis",
            prediction_time.strftime("%Y%m%d_%H%M%S")
        )
        ensure_dir(save_dir)
        
        # 원본 데이터 시각화
        plot_sensor_data(
            df, 
            cols=[target_column], 
            save_path=os.path.join(save_dir, "sensor_data.png")
        )
        
        # 예측 결과 시각화
        plot_resource_prediction(
            df, 
            target_column, 
            results, 
            save_path=os.path.join(save_dir, "resource_prediction_results.png")
        )
        
        # 장기 예측 시각화
        if scenario_results:
            # 시각화 유틸리티 모듈 사용
            plot_long_term_scenarios(
                scenario_results,
                capacity_threshold=config["resource_analysis"]["capacity_threshold"],
                expansion_dates=expansion_dates,
                save_path=os.path.join(save_dir, "long_term_scenarios.png")
            )
        
        # 학습 정보 저장
        training_info = model.get_training_info()
        with open(os.path.join(save_dir, "training_info.json"), 'w') as f:
            json.dump(training_info, f, indent=4)
    
    # CSV 파일로 저장
    if config["results"]["save_csv"]:
        csv_dir = os.path.join(
            config["results"]["save_dir"],
            "resource_analysis",
            "csv"
        )
        ensure_dir(csv_dir)
        
        # 단기 예측 저장
        result_df = results.reset_index() if isinstance(results.index, pd.DatetimeIndex) else results
        csv_path = os.path.join(
            csv_dir,
            f"resource_predictions_{prediction_time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        result_df.to_csv(csv_path, index=False)
        logger.info(f"예측 결과 CSV 저장: {csv_path}")
        
        # 장기 예측 저장
        if scenario_results:
            for scenario, forecast in scenario_results.items():
                scenario_path = os.path.join(
                    csv_dir,
                    f"resource_forecast_{scenario}_{prediction_time.strftime('%Y%m%d_%H%M%S')}.csv"
                )
                forecast.to_csv(scenario_path)
                logger.info(f"장기 예측 결과 CSV 저장: {scenario_path}")
    
    # 타겟 컬럼이 데이터에 있는지 확인 및 대체
    target_column = resource_config["target_column"]
    if target_column not in df.columns:
        logger.error(f"타겟 컬럼 '{target_column}'이 데이터에 없습니다.")
        logger.info(f"사용 가능한 컬럼: {df.columns.tolist()}")
        
        # 대체 컬럼 찾기 - 자원 사용량 후보
        target_candidates = [
            'user', 'usage_user',                   # CPU 사용자 사용률
            'system', 'usage_system',               # CPU 시스템 사용률
            'mem_usedPercent_used_percent',        # 메모리 사용률
            'disk_usedPercent_used_percent',       # 디스크 사용률
            'cpu_total_usage',                     # CPU 총 사용률 (생성된 특성)
            'load1',                              # 시스템 로드
            'used_percent'                         # 일반 사용률
        ]
        
        for candidate in target_candidates:
            if candidate in df.columns:
                logger.info(f"'{candidate}' 컬럼을 타겟으로 사용합니다.")
                resource_config["target_column"] = candidate
                break
        else:
            logger.error("적절한 타겟 컬럼을 찾을 수 없습니다.")
            return False, None
    
    # 업데이트된 타겟 컬럼 사용
    target_column = resource_config["target_column"]
    
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
    resource_config = config["resource_analysis"]
    
    # 충분한 데이터가 있는지 확인
    min_required_samples = 2  # 최소한의 학습 가능 데이터 수
    if len(df) < min_required_samples:
        logger.error(f"데이터가 너무 적습니다: {len(df)}행. 최소 {min_required_samples}행 필요합니다.")
        return False, None
    
    # 고급 특성 생성 (추가된 부분)
    df = _create_advanced_features(df)
    
    # 예측 기간 및 입력 윈도우 설정
    pred_horizon = 24  # 24시간 예측
    input_window = min(resource_config["input_window"], len(df) - pred_horizon)
    if input_window <= 0:
        input_window = 1  # 최소값 설정
        
    # 모델 생성
    input_dim = df.shape[1]
    model = ResourceLSTM(
        input_dim=input_dim,
        input_window=input_window,
        pred_horizon=pred_horizon,
        model_save_path=os.path.join(model_dir, 'resource_lstm.h5')
    )
    
    # 모델 학습
    logger.info("자원 사용량 예측 모델 학습 중...")
    
    history = model.fit(
        df, 
        target_col=target_column,
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
    logger.info("자원 사용량 예측 수행 중...")
    try:
        results = model.predict(df, target_column)
        
        # 결과가 비어있는지 확인
        if results.empty:
            logger.warning("예측 결과가 비어 있습니다.")
            return False, None
        
        # 디바이스 ID 및 예측 시간 추가
        results_df = results.reset_index()
        results_df.rename(columns={'index': 'timestamp'}, inplace=True)
        results_df['device_id'] = "device_001"  # 기본값
        results_df['resource_type'] = target_column
        results_df['prediction_time'] = datetime.now()
        
        # 결과 저장 (MySQL)
        if 'mysql' in config:
            try:
                save_to_mysql(results_df, "resource_predictions", config)
            except Exception as e:
                logger.error(f"MySQL 저장 중 오류: {e}")
                # MySQL 오류가 발생해도 계속 진행
        
        # 장기 예측 수행
        logger.info("자원 사용량 장기 예측 수행 중...")
        scenario_results = {}
        expansion_dates = {}
        
        try:
            scenario_results, expansion_dates = model.predict_long_term(
                periods=180,  # 6개월
                capacity_threshold=resource_config["capacity_threshold"]
            )
            
            # 용량 계획 결과 저장 (MySQL)
            if scenario_results and 'mysql' in config:
                try:
                    capacity_df = pd.DataFrame([
                        {
                            'scenario': scenario,
                            'resource_type': target_column,
                            'expansion_date': date.strftime('%Y-%m-%d') if date else None,
                            'threshold_value': resource_config["capacity_threshold"],
                            'prediction_time': datetime.now()
                        }
                        for scenario, date in expansion_dates.items()
                    ])
                    
                    save_to_mysql(capacity_df, "capacity_planning", config)
                except Exception as e:
                    logger.error(f"용량 계획 결과 저장 중 오류: {e}")
        except Exception as e:
            logger.error(f"장기 예측 실패: {e}")
        
        # 결과 시각화 및 저장
        _save_prediction_results(config, df, target_column, results, scenario_results, 
                                expansion_dates, datetime.now(), model)
        
        logger.info("자원 사용량 분석 완료")
        for scenario, date in expansion_dates.items():
            if date:
                logger.info(f"- 시나리오 '{scenario}': 용량 증설 필요 시점: {date.strftime('%Y-%m-%d')}")
            else:
                logger.info(f"- 시나리오 '{scenario}': 예측 기간 내 용량 증설 필요 없음")
        
        return True, results
        
    except Exception as e:
        logger.error(f"자원 사용량 예측 중 오류 발생: {e}")
        return False, None