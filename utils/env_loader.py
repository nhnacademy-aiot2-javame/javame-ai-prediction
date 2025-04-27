#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
환경 변수 로드 유틸리티

.env 파일 또는 시스템 환경 변수에서 설정을 로드하고 config.json 업데이트
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# 로깅 설정
logger = logging.getLogger(__name__)

def load_environment_variables():
    """
    환경 변수 로드 및 기본값 설정
    
    Returns:
        dict: 환경 변수 딕셔너리
    """
    # .env 파일 로드 (있는 경우)
    load_dotenv()
    
    # 환경 변수 딕셔너리
    env_vars = {
        # InfluxDB 설정
        "INFLUXDB_URL": os.environ.get("INFLUXDB_URL", "http://influxdb.javame.live"),
        "INFLUXDB_TOKEN": os.environ.get("INFLUXDB_TOKEN", "g-W7W0j9AE4coriQfnhHGMDnDhTZGok8bgY1NnZ6Z0EnTOsFY3SWAqDTC5fYlQ9mYnbK_doR074-a4Dgck2AOQ=="),
        "INFLUXDB_ORG": os.environ.get("INFLUXDB_ORG", "javame"),
        "INFLUXDB_BUCKET": os.environ.get("INFLUXDB_BUCKET", "data"),
        
        # MySQL 설정
        "MYSQL_HOST": os.environ.get("MYSQL_HOST", "s4.java21.net"),
        "MYSQL_PORT": int(os.environ.get("MYSQL_PORT", 18080)),
        "MYSQL_USER": os.environ.get("MYSQL_USER", "aiot02_team3"),
        "MYSQL_PASSWORD": os.environ.get("MYSQL_PASSWORD", "ryL7LcSp@Yiz[bR7"),
        "MYSQL_DATABASE": os.environ.get("MYSQL_DATABASE", "aiot02_team3"),
        
        # 애플리케이션 설정
        "PREDICTION_INTERVAL": int(os.environ.get("PREDICTION_INTERVAL", 6)),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
        "TEST_MODE": os.environ.get("TEST_MODE", "false").lower() == "true"
    }
    
    logger.info("환경 변수 로드 완료")
    return env_vars

def update_config_from_env(config_path="config.json"):
    """
    환경 변수를 기반으로 config.json 파일 업데이트
    
    Args:
        config_path (str): config.json 파일 경로
        
    Returns:
        dict: 업데이트된 설정 딕셔너리
    """
    # 환경 변수 로드
    env_vars = load_environment_variables()
    
    # config.json 파일 확인
    config_file = Path(config_path)
    config = {}
    
    # 파일이 존재하면 로드
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"기존 설정 파일 로드: {config_path}")
        except Exception as e:
            logger.error(f"설정 파일 로드 중 오류 발생: {e}")
    
    # 환경 변수로 설정 업데이트
    # 1. InfluxDB 설정
    if "influxdb" not in config:
        config["influxdb"] = {}
    
    config["influxdb"]["url"] = env_vars["INFLUXDB_URL"]
    config["influxdb"]["token"] = env_vars["INFLUXDB_TOKEN"]
    config["influxdb"]["org"] = env_vars["INFLUXDB_ORG"]
    config["influxdb"]["bucket"] = env_vars["INFLUXDB_BUCKET"]
    
    # 여러 measurement 지원
    # INFLUXDB_MEASUREMENTS 환경 변수가 있으면 사용 (콤마로 구분된 목록)
    measurements_env = os.environ.get("INFLUXDB_MEASUREMENTS", "")
    # if measurements_env:
    #     config["influxdb"]["measurements"] = [m.strip() for m in measurements_env.split(",")]
    # # 아니면 기본 measurement 배열 설정
    # elif "measurements" not in config["influxdb"]:
    #     # 기존 단일 measurement가 있으면 그것을 배열에 포함
    #     if "measurement" in config["influxdb"]:
    #         config["influxdb"]["measurements"] = [config["influxdb"]["measurement"]]
    #     else:
    #         config["influxdb"]["measurements"] = ["system", "cpu", "memory", "disk", "sensors"]
    
    # # 하위 호환성을 위해 첫 번째 measurement를 단일 measurement 필드로 설정
    # if "measurements" in config["influxdb"] and config["influxdb"]["measurements"]:
    #     config["influxdb"]["measurement"] = config["influxdb"]["measurements"][0]
    
    # 2. MySQL 설정
    if "mysql" not in config:
        config["mysql"] = {}
    
    config["mysql"]["host"] = env_vars["MYSQL_HOST"]
    config["mysql"]["port"] = env_vars["MYSQL_PORT"]
    config["mysql"]["user"] = env_vars["MYSQL_USER"]
    config["mysql"]["password"] = env_vars["MYSQL_PASSWORD"]
    config["mysql"]["database"] = env_vars["MYSQL_DATABASE"]
    
    # 3. 예측 설정
    if "failure_prediction" not in config:
        config["failure_prediction"] = {
            "enabled": True,
            "interval_hours": env_vars["PREDICTION_INTERVAL"],
            "input_window": 24,
            "target_column": "temp_input",
            "threshold": 0.75
        }
    else:
        config["failure_prediction"]["interval_hours"] = env_vars["PREDICTION_INTERVAL"]
    
    if "resource_analysis" not in config:
        config["resource_analysis"] = {
            "enabled": True,
            "interval_hours": env_vars["PREDICTION_INTERVAL"] * 2,
            "input_window": 48,
            "target_column": "user",
            "capacity_threshold": 80
        }
    else:
        config["resource_analysis"]["interval_hours"] = env_vars["PREDICTION_INTERVAL"] * 2
    
    # 4. 결과 저장 설정
    if "results" not in config:
        config["results"] = {
            "save_dir": "prediction_results",
            "save_plots": True,
            "save_csv": True,
            "model_dir": "model_weights",
            "log_dir": "logs"
        }
    
    # 5. 로깅 설정
    if "logging" not in config:
        config["logging"] = {
            "level": env_vars["LOG_LEVEL"],
            "log_file": "logs/iot_prediction.log",
            "training_log_dir": "logs/training_logs",
            "max_size_mb": 10,
            "backup_count": 5
        }
    else:
        config["logging"]["level"] = env_vars["LOG_LEVEL"]
    
    # 6. 테스트 설정
    if "test" not in config:
        config["test"] = {
            "enabled": env_vars["TEST_MODE"],
            "days": 30,
            "hour_interval": 1,
            "device_count": 1
        }
    else:
        config["test"]["enabled"] = env_vars["TEST_MODE"]
    
    # 업데이트된 설정 저장
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"설정 파일 업데이트 완료: {config_path}")
    except Exception as e:
        logger.error(f"설정 파일 저장 중 오류 발생: {e}")
    
    return config

if __name__ == "__main__":
    # 직접 실행 시 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # config.json 업데이트
    update_config_from_env()