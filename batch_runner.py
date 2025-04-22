#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IoT 예측 시스템 배치 실행기

Docker 환경에서 주기적으로 예측 작업을 실행하는 스크립트
환경 변수를 통해 설정을 관리하고 실패 시 재시도 로직 포함
"""

import os
import sys
import json
import time
import signal
import logging
import logging.handlers
import traceback
from datetime import datetime, timedelta
import schedule

# 사용자 정의 모듈 임포트
from utils.env_loader import update_config_from_env
from utils.batch_monitor import monitor_batch_job

logger = logging.getLogger(__name__)

# 환경 변수를 통한 설정
INFLUXDB_URL = os.environ.get('INFLUXDB_URL', 'http://localhost:10288')
INFLUXDB_TOKEN = os.environ.get('INFLUXDB_TOKEN', 'g-W7W0j9AE4coriQfnhHGMDnDhTZGok8bgY1NnZ6Z0EnTOsFY3SWAqDTC5fYlQ9mYnbK_doR074-a4Dgck2AOQ==')
INFLUXDB_ORG = os.environ.get('INFLUXDB_ORG', 'javame')
INFLUXDB_BUCKET = os.environ.get('INFLUXDB_BUCKET', 'aiot')
MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 18080))
MYSQL_USER = os.environ.get('MYSQL_USER', 'aiot02_team3')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'ryL7LcSp@Yiz[bR7')
MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE', 'aiot02_team3')
PREDICTION_INTERVAL = int(os.environ.get('PREDICTION_INTERVAL', 6))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# 로깅 설정
def setup_logging():
    log_level = getattr(logging, LOG_LEVEL)
    
    # 로그 디렉토리 생성
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 핸들러 설정
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/batch_runner.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# config.json 동적 생성 또는 업데이트
def create_or_update_config():
    config_path = 'config.json'
    
    # 기본 설정
    config = {
        "influxdb": {
            "url": INFLUXDB_URL,
            "token": INFLUXDB_TOKEN,
            "org": INFLUXDB_ORG,
            "bucket": INFLUXDB_BUCKET
        },
        "mysql": {
            "host": MYSQL_HOST,
            "port": MYSQL_PORT,
            "user": MYSQL_USER,
            "password": MYSQL_PASSWORD,
            "database": MYSQL_DATABASE
        },
        "failure_prediction": {
            "enabled": True,
            "interval_hours": PREDICTION_INTERVAL,
            "input_window": 24,
            "target_column": "temp_input",
            "threshold": 0.75
        },
        "resource_analysis": {
            "enabled": True,
            "interval_hours": PREDICTION_INTERVAL * 2,  # 실패 예측보다 덜 자주 실행
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
            "level": LOG_LEVEL,
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
        }
    }
    
    # 기존 설정 파일이 있는 경우, 기본값과 병합
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            
            # 환경 변수로 설정된 주요 값 업데이트
            if "influxdb" in existing_config:
                existing_config["influxdb"]["url"] = INFLUXDB_URL
                existing_config["influxdb"]["token"] = INFLUXDB_TOKEN
                existing_config["influxdb"]["org"] = INFLUXDB_ORG
                existing_config["influxdb"]["bucket"] = INFLUXDB_BUCKET
            
            if "mysql" in existing_config:
                existing_config["mysql"]["host"] = MYSQL_HOST
                existing_config["mysql"]["port"] = MYSQL_PORT
                existing_config["mysql"]["user"] = MYSQL_USER
                existing_config["mysql"]["password"] = MYSQL_PASSWORD
                existing_config["mysql"]["database"] = MYSQL_DATABASE
            
            # 기존 설정 유지하되 환경 변수로 설정된 값 우선
            config = existing_config
        except Exception as e:
            logger.error(f"기존 config.json 파일 로드 중 오류 발생: {e}")
    
    # 설정 파일 저장
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"설정 파일 생성/업데이트 완료: {config_path}")
    except Exception as e:
        logger.error(f"설정 파일 저장 중 오류 발생: {e}")

# 예측 작업 실행 함수
@monitor_batch_job(json.load(open('config.json')) if os.path.exists('config.json') else {"mysql": {}})
def run_prediction_job(mode='all', retry_count=3, retry_delay=60):
    """
    예측 작업을 실행하고 실패 시 재시도
    
    Args:
        mode (str): 실행 모드 ('all', 'failure', 'resource')
        retry_count (int): 실패 시 재시도 횟수
        retry_delay (int): 재시도 간 대기 시간(초)
        
    Returns:
        bool: 예측 작업 성공 여부
    """
    logger.info(f"===== 예측 작업 시작 (모드: {mode}) =====")
    
    for attempt in range(retry_count + 1):
        try:
            # 실행 명령 구성
            cmd = f"python run_prediction.py --mode {mode} --config config.json --schedule"
            
            # 작업 실행
            logger.info(f"예측 작업 실행 중 (시도 {attempt + 1}/{retry_count + 1}): {cmd}")
            exit_code = os.system(cmd)
            
            if exit_code == 0:
                logger.info(f"예측 작업 성공 완료 (모드: {mode})")
                return True
            else:
                logger.warning(f"예측 작업 실패 (종료 코드: {exit_code})")
                
                if attempt < retry_count:
                    logger.info(f"{retry_delay}초 후 재시도...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"최대 재시도 횟수 초과. 작업 실패 (모드: {mode})")
                    return False
        
        except Exception as e:
            logger.error(f"예측 작업 실행 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
            
            if attempt < retry_count:
                logger.info(f"{retry_delay}초 후 재시도...")
                time.sleep(retry_delay)
            else:
                logger.error(f"최대 재시도 횟수 초과. 작업 실패 (모드: {mode})")
                return False

# 데이터베이스 연결 테스트
def test_database_connections():
    """
    데이터베이스 연결 테스트 실행
    
    Returns:
        bool: 연결 성공 여부
    """
    logger.info("데이터베이스 연결 테스트 중...")
    
    try:
        # db_connection_test.py 실행
        exit_code = os.system("python3 db_connection_test.py config.json")
        
        if exit_code == 0:
            logger.info("데이터베이스 연결 테스트 성공")
            return True
        else:
            logger.error(f"데이터베이스 연결 테스트 실패 (종료 코드: {exit_code})")
            return False
    
    except Exception as e:
        logger.error(f"데이터베이스 연결 테스트 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return False

# 예측 작업 스케줄링
def schedule_prediction_jobs():
    """예측 작업 스케줄링"""
    # 고장 예측 스케줄 설정
    schedule.every(PREDICTION_INTERVAL).hours.do(
        run_prediction_job, mode='failure'
    )
    logger.info(f"고장 예측 작업이 {PREDICTION_INTERVAL}시간마다 실행되도록 예약되었습니다.")
    
    # 자원 사용량 분석 스케줄 설정
    schedule.every(PREDICTION_INTERVAL * 2).hours.do(
        run_prediction_job, mode='resource'
    )
    logger.info(f"자원 사용량 분석이 {PREDICTION_INTERVAL * 2}시간마다 실행되도록 예약되었습니다.")
    
    # 처음 한 번은 바로 실행
    logger.info("초기 예측 작업 실행 중...")
    run_prediction_job(mode='all')

# 종료 핸들러
def handle_exit(signum, frame):
    """프로세스 종료 처리"""
    logger.info("종료 신호 수신. 실행 중인 작업 정리 중...")
    # 여기에 필요한 정리 작업 추가
    sys.exit(0)

# 메인 함수
def main():
    """메인 함수"""
    # 종료 신호 핸들러 등록
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)
    
    logger.info("===== IoT 예측 시스템 배치 러너 시작 =====")
    
    # 설정 파일 생성 또는 업데이트
    config = create_or_update_config()
    
    # 필요한 디렉토리 생성
    for directory in ['logs', 'model_weights', 'prediction_results']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"디렉토리 생성: {directory}")
    
    # 데이터베이스 연결 테스트
    if not test_database_connections():
        logger.warning("데이터베이스 연결 테스트 실패. 계속 진행합니다.")
    
    # 예측 간격 확인
    prediction_interval = config.get('failure_prediction', {}).get('interval_hours', 6)
    resource_interval = config.get('resource_analysis', {}).get('interval_hours', 12)
    
    # 예측 작업 스케줄링
    logger.info(f"고장 예측 작업 스케줄링: {prediction_interval}시간마다")
    schedule.every(prediction_interval).hours.do(
        run_prediction_job, mode='failure'
    )
    
    logger.info(f"자원 사용량 분석 스케줄링: {resource_interval}시간마다")
    schedule.every(resource_interval).hours.do(
        run_prediction_job, mode='resource'
    )
    
    # 처음 한 번은 바로 실행
    logger.info("초기 예측 작업 실행 중...")
    run_prediction_job(mode='all')
    
    # 스케줄러 실행 루프
    logger.info("스케줄러 실행 중...")
    
    try:
        # 다음 작업 시간 표시
        show_next_scheduled_runs()
        
        while True:
            schedule.run_pending()
            time.sleep(1)
            
            # 매시간 다음 작업 시간 업데이트
            if datetime.now().minute == 0 and datetime.now().second == 0:
                show_next_scheduled_runs()
                
    except Exception as e:
        logger.error(f"스케줄러 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("===== IoT 예측 시스템 배치 러너 종료 =====")

def show_next_scheduled_runs():
    """다음 예약된 작업 표시"""
    logger.info("다음 예약된 작업:")
    for job in schedule.jobs:
        next_run = job.next_run
        if next_run:
            time_diff = next_run - datetime.now()
            hours, remainder = divmod(time_diff.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(f"- {job.job_func.__name__}: {int(hours)}시간 {int(minutes)}분 후 ({next_run.strftime('%Y-%m-%d %H:%M:%S')})")

if __name__ == "__main__":
    logger = setup_logging()
    main()