#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
설정 관리 유틸리티 모듈

설정 파일 로드, 유효성 검사, 동적 설정 조정 등 기능 제공
"""

import os
import json
import logging
from datetime import datetime

# 로깅 설정
logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_CONFIG = {
    "influxdb": {
        "url": "http://localhost:8086",
        "token": "your-token",
        "org": "your-org",
        "bucket": "iot_sensors",
        "measurement": "system"
    },
    "mysql": {
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "password",
        "database": "iot_predictions"
    },
    "failure_prediction": {
        "enabled": True,
        "interval_hours": 1,
        "input_window": 24,
        "target_column": "temp_input",
        "threshold": 0.75
    },
    "resource_analysis": {
        "enabled": True,
        "interval_hours": 12,
        "input_window": 48,
        "target_column": "user",
        "capacity_threshold": 80
    },
    "results": {
        "save_dir": "prediction_results",
        "save_plots": True,
        "save_csv": True,
        "model_dir": "model_weights",
        "log_dir": "logs"
    },
    "logging": {
        "level": "INFO",
        "log_file": "logs/iot_prediction.log",
        "training_log_dir": "logs/training_logs",
        "max_size_mb": 10,
        "backup_count": 5
    },
    "test": {
        "enabled": False,
        "days": 30,
        "hour_interval": 1,
        "device_count": 1
    },
    "advanced": {
        "feature_engineering": True,
        "outlier_removal": True,
        "use_correlation_features": True,
        "normalization_method": "minmax",
        "validation_split": 0.2,
        "early_stopping_patience": 10
    }
}

def load_config(config_path=None):
    """
    설정 파일 로드
    
    Args:
        config_path (str): 설정 파일 경로 (None이면 기본 설정 반환)
        
    Returns:
        dict: 설정 정보
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
            # 사용자 설정으로 기본 설정 업데이트 (재귀적)
            config = _update_nested_dict(config, user_config)
                    
            logger.info(f"설정 파일을 로드했습니다: {config_path}")
        except Exception as e:
            logger.error(f"설정 파일 로드 중 오류 발생: {e}")
    else:
        logger.info("설정 파일이 지정되지 않았거나 존재하지 않아 기본 설정을 사용합니다.")
    
    # 설정 유효성 검사
    validate_config(config)
    
    return config

def _update_nested_dict(d, u):
    """
    중첩 딕셔너리 업데이트 (재귀적)
    
    Args:
        d (dict): 기본 딕셔너리
        u (dict): 업데이트할 딕셔너리
        
    Returns:
        dict: 업데이트된 딕셔너리
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = _update_nested_dict(d[k].copy(), v)
        else:
            d[k] = v
    return d

def validate_config(config):
    """
    설정의 유효성 검사 및 자동 수정
    
    Args:
        config (dict): 검사할 설정
        
    Returns:
        dict: 검증 및 수정된 설정
    """
    # 필수 섹션 확인
    required_sections = ["influxdb", "failure_prediction", "resource_analysis", "results", "logging"]
    for section in required_sections:
        if section not in config:
            logger.warning(f"필수 설정 섹션 '{section}'이 누락되었습니다. 기본값을 사용합니다.")
            config[section] = DEFAULT_CONFIG[section]
    
    # InfluxDB 설정 검증
    if "influxdb" in config:
        influx_config = config["influxdb"]
        required_influx_keys = ["url", "token", "org", "bucket", "measurement"]
        
        for key in required_influx_keys:
            if key not in influx_config:
                logger.warning(f"InfluxDB 설정에서 '{key}'가 누락되었습니다. 기본값을 사용합니다.")
                influx_config[key] = DEFAULT_CONFIG["influxdb"][key]
    
    # MySQL 설정 검증
    if "mysql" in config:
        mysql_config = config["mysql"]
        required_mysql_keys = ["host", "user", "password", "database"]
        
        for key in required_mysql_keys:
            if key not in mysql_config:
                logger.warning(f"MySQL 설정에서 '{key}'가 누락되었습니다. 기본값을 사용합니다.")
                mysql_config[key] = DEFAULT_CONFIG["mysql"][key]
        
        # 포트 검증
        if "port" not in mysql_config:
            mysql_config["port"] = 3306
            logger.info("MySQL 포트가 지정되지 않아 기본값(3306)을 사용합니다.")
    
    # 예측 설정 검증
    pred_sections = ["failure_prediction", "resource_analysis"]
    for section in pred_sections:
        if section in config:
            pred_config = config[section]
            
            # 기본값 확인
            required_pred_keys = ["enabled", "interval_hours", "input_window", "target_column"]
            for key in required_pred_keys:
                if key not in pred_config:
                    logger.warning(f"{section} 설정에서 '{key}'가 누락되었습니다. 기본값을 사용합니다.")
                    pred_config[key] = DEFAULT_CONFIG[section][key]
            
            # 값 범위 검증
            if pred_config["interval_hours"] < 1:
                logger.warning(f"{section} 간격이 너무 짧습니다. 최소 1시간으로 설정합니다.")
                pred_config["interval_hours"] = 1
            
            if pred_config["input_window"] < 2:
                logger.warning(f"{section} 입력 윈도우가 너무 작습니다. 최소 2로 설정합니다.")
                pred_config["input_window"] = 2
    
    # 결과 설정 검증
    if "results" in config:
        results_config = config["results"]
        if "save_dir" not in results_config:
            results_config["save_dir"] = "prediction_results"
        
        # 디렉토리 경로 정규화
        for dir_key in ["save_dir", "model_dir", "log_dir"]:
            if dir_key in results_config:
                # 상대 경로를 절대 경로로 변환하지 않음
                pass
    
    # 로깅 설정 검증
    if "logging" in config:
        logging_config = config["logging"]
        if "level" in logging_config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if logging_config["level"] not in valid_levels:
                logger.warning(f"유효하지 않은 로깅 레벨: {logging_config['level']}. 'INFO'로 설정합니다.")
                logging_config["level"] = "INFO"
    
    # 고급 설정 검증
    if "advanced" not in config:
        logger.info("고급 설정이 없습니다. 기본값을 사용합니다.")
        config["advanced"] = DEFAULT_CONFIG["advanced"]
    
    return config

def save_config(config, config_path=None):
    """
    설정을 파일로 저장
    
    Args:
        config (dict): 저장할 설정
        config_path (str): 저장 경로 (None이면 config.json)
        
    Returns:
        bool: 성공 여부
    """
    if config_path is None:
        config_path = "config.json"
    
    try:
        # 백업 파일 생성
        if os.path.exists(config_path):
            backup_path = f"config.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_path, 'w', encoding='utf-8') as f:
                with open(config_path, 'r', encoding='utf-8') as orig:
                    f.write(orig.read())
            logger.info(f"기존 설정 백업 완료: {backup_path}")
        
        # 새 설정 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        logger.info(f"설정 저장 완료: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"설정 저장 중 오류 발생: {e}")
        return False

def update_config_for_column(config, column_name, column_type=None):
    """
    특정 컬럼에 맞게 설정 업데이트
    
    Args:
        config (dict): 설정 정보
        column_name (str): 컬럼 이름
        column_type (str): 컬럼 유형 ('cpu', 'memory', 'disk', 'temp', 'network', 'system')
        
    Returns:
        dict: 업데이트된 설정
    """
    updated_config = config.copy()
    
    # 컬럼 유형 자동 감지
    if column_type is None:
        column_type = _detect_column_type(column_name)
    
    # 고장 예측 설정 업데이트
    if column_type == 'temp':
        # 온도 컬럼은 고장 예측에 적합
        updated_config["failure_prediction"]["target_column"] = column_name
        updated_config["failure_prediction"]["threshold"] = 0.8  # 온도 기반 임계값
        logger.info(f"고장 예측 타겟을 '{column_name}'으로 설정했습니다.")
    elif column_type in ['cpu', 'system']:
        # CPU/시스템 컬럼도 고장 예측에 사용 가능
        updated_config["failure_prediction"]["target_column"] = column_name
        updated_config["failure_prediction"]["threshold"] = 0.75
        logger.info(f"고장 예측 타겟을 '{column_name}'으로 설정했습니다.")
    
    # 자원 사용량 분석 설정 업데이트
    if column_type in ['cpu', 'memory', 'disk']:
        # 자원 사용량 컬럼
        updated_config["resource_analysis"]["target_column"] = column_name
        
        # 컬럼 유형별 임계값 조정
        if column_type == 'cpu':
            updated_config["resource_analysis"]["capacity_threshold"] = 80
        elif column_type == 'memory':
            updated_config["resource_analysis"]["capacity_threshold"] = 90
        elif column_type == 'disk':
            updated_config["resource_analysis"]["capacity_threshold"] = 85
        
        logger.info(f"자원 사용량 분석 타겟을 '{column_name}'으로 설정했습니다.")
    
    return updated_config

def _detect_column_type(column_name):
    """
    컬럼 이름에서 유형 감지
    
    Args:
        column_name (str): 컬럼 이름
        
    Returns:
        str: 컬럼 유형 ('cpu', 'memory', 'disk', 'temp', 'network', 'system', 'unknown')
    """
    column_lower = column_name.lower()
    
    if any(kw in column_lower for kw in ['cpu', 'usage_user', 'usage_system', 'usage_idle', 'usage_iowait']):
        return 'cpu'
    elif any(kw in column_lower for kw in ['mem', 'memory', 'used_percent', 'available']):
        return 'memory'
    elif any(kw in column_lower for kw in ['disk', 'io', 'read', 'write']):
        return 'disk'
    elif any(kw in column_lower for kw in ['temp', 'temperature']):
        return 'temp'
    elif any(kw in column_lower for kw in ['net', 'bytes_sent', 'bytes_recv', 'drop', 'err']):
        return 'network'
    elif any(kw in column_lower for kw in ['load', 'system', 'uptime']):
        return 'system'
    else:
        return 'unknown'

def get_optimal_config_for_dataset(config, df):
    """
    데이터셋에 최적화된 설정 생성
    
    Args:
        config (dict): 기본 설정
        df (pd.DataFrame): 데이터프레임
        
    Returns:
        dict: 최적화된 설정
    """
    if df is None or df.empty:
        logger.warning("데이터가 없어 기본 설정을 반환합니다.")
        return config.copy()
    
    optimal_config = config.copy()
    
    try:
        # 1. 데이터 크기에 맞는 설정
        data_length = len(df)
        
        # 입력 윈도우 조정 (데이터 길이의 1/4, 최소 5, 최대 48)
        window_size = max(5, min(data_length // 4, 48))
        optimal_config["failure_prediction"]["input_window"] = window_size
        optimal_config["resource_analysis"]["input_window"] = window_size
        
        logger.info(f"데이터 길이({data_length})에 맞게 입력 윈도우를 {window_size}로 조정했습니다.")
        
        # 2. 컬럼 특성에 따른 타겟 선택
        # CPU 관련 컬럼 확인
        cpu_cols = [col for col in df.columns if any(kw in col.lower() for kw in 
                                                  ['usage_user', 'usage_system', 'usage_idle', 'usage_iowait'])]
        if cpu_cols:
            # CPU 사용률을 자원 분석 타겟으로 사용
            user_col = next((col for col in cpu_cols if 'user' in col.lower()), None)
            if user_col:
                optimal_config["resource_analysis"]["target_column"] = user_col
                logger.info(f"자원 분석 타겟으로 CPU 사용률 컬럼 '{user_col}'을 선택했습니다.")
        
        # 온도 컬럼 확인
        temp_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['temp', 'temperature'])]
        if temp_cols:
            # 온도를 고장 예측 타겟으로 사용
            optimal_config["failure_prediction"]["target_column"] = temp_cols[0]
            logger.info(f"고장 예측 타겟으로 온도 컬럼 '{temp_cols[0]}'을 선택했습니다.")
        
        # 3. 모델 하이퍼파라미터 최적화 (고급 설정)
        if "advanced" not in optimal_config:
            optimal_config["advanced"] = {}
        
        # 데이터 크기에 따른 배치 크기 조정
        if data_length < 100:
            optimal_config["advanced"]["batch_size"] = 8
        elif data_length < 1000:
            optimal_config["advanced"]["batch_size"] = 16
        else:
            optimal_config["advanced"]["batch_size"] = 32
        
        # 에포크 수 조정
        if data_length < 100:
            optimal_config["advanced"]["epochs"] = 30
        elif data_length < 1000:
            optimal_config["advanced"]["epochs"] = 50
        else:
            optimal_config["advanced"]["epochs"] = 100
        
        # 검증 분할 조정
        if data_length < 50:
            optimal_config["advanced"]["validation_split"] = 0.1
        else:
            optimal_config["advanced"]["validation_split"] = 0.2
        
        logger.info(f"데이터셋에 최적화된 설정을 생성했습니다.")
        return optimal_config
        
    except Exception as e:
        logger.error(f"최적 설정 생성 중 오류 발생: {e}")
        return config.copy()