#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IoT 예측 시스템 메인 실행 파일 (리팩토링 버전)

두 가지 기능 구현:
1. 고장 예측 (Failure Prediction): 센서 데이터를 분석하여 설비 고장 예측
2. 자원 사용량 분석 (Resource Usage Analysis): 자원 사용량 예측 및 용량 증설 계획

사용법:
    python run_prediction.py [--mode 모드] [--config 설정파일] [--option 옵션]
"""

import os
import sys
import json
import argparse
import logging
import schedule
import time
from datetime import datetime, timedelta
import logging.handlers

# 사용자 정의 모듈 임포트
from utils.data_loader import load_all_data, query_influxdb, generate_test_data
from utils.db_utils import test_influxdb_connection, test_mysql_connection, init_mysql
from utils.config_utils import load_config, get_optimal_config_for_dataset
from predictors.failure_predictor import run_failure_prediction
from predictors.resource_predictor import run_resource_analysis

# 데이터 파일 경로 정의 (data 폴더 내부)
DATA_FILES = {
    'cpu': [
        'data/cpu_usageUser.csv',
        'data/cpu_usageSystem.csv',
        'data/cpu_usageIowait.csv',
        'data/cpu_usageIdle.csv'
    ],
    'memory': [
        'data/mem_usedPercent.csv',
        'data/mem_availablePercent.csv'
    ],
    'disk': [
        'data/disk_usedPercent.csv',
        'data/diskio_data.csv'
    ],
    'sensors': [
        'data/sensors_tempInput.csv'
    ],
    'system': [
        'data/system_data.csv'
    ],
    'network': [  # 새로 추가된 네트워크 데이터
        'data/net_bytes_recv.csv',
        'data/net_bytes_sent.csv',
        'data/net_drop_in.csv',
        'data/net_drop_out.csv'
    ]
}

# 로깅 설정 함수
def setup_logging(config):
    """로깅 설정"""
    log_config = config.get("logging", {})
    log_file = log_config.get("log_file", "logs/iot_prediction.log")
    log_dir = os.path.dirname(log_file)
    
    # 로그 디렉토리 생성
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 핸들러 설정
    handlers = []
    
    # 파일 핸들러 - 회전식 로그
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_config.get("max_size_mb", 10) * 1024 * 1024,  # MB를 바이트로 변환
        backupCount=log_config.get("backup_count", 5)
    )
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    handlers.append(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    handlers.append(console_handler)
    
    # 로거 설정
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def ensure_dir(directory):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"디렉토리 생성: {directory}")

def scheduled_failure_prediction(config):
    """고장 예측 스케줄링 함수"""
    try:
        # 데이터 로드
        logger.info("자원 사용량 분석을 위한 데이터 로드 중...")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=config["resource_analysis"]["input_window"] * 2)

        df = query_influxdb(config, start_time)

        if df is None or df.empty:
            logger.error("데이터를 로드할 수 없습니다. 고장 예측을 건너뜁니다.")
            return
        else:
            logger.info("데이터 로드 완료.")
        
            
        # 설정 최적화 (선택 사항)
        if config.get("advanced", {}).get("optimize_config", False):
            logger.info("데이터셋에 맞게 설정 최적화 중...")
            optimized_config = get_optimal_config_for_dataset(config, df)
            if optimized_config:
                config = optimized_config
                logger.info("설정이 최적화되었습니다.")
        
        # 고장 예측 실행
        success, results = run_failure_prediction(config, df)
        
        if success:
            logger.info("고장 예측이 성공적으로 완료되었습니다.")
            # 필요한 경우 여기에 추가 처리 작업 수행
        else:
            logger.warning("고장 예측이 실패했거나 결과가 없습니다.")
    
    except Exception as e:
        logger.error(f"고장 예측 중 오류 발생: {e}", exc_info=True)

def scheduled_resource_analysis(config):
    """자원 사용량 분석 스케줄링 함수"""
    try:
        # 데이터 로드
        logger.info("자원 사용량 분석을 위한 데이터 로드 중...")
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=config["resource_analysis"]["input_window"] * 2)

        df = query_influxdb(config, start_time)

        if df is None or df.empty:
            logger.error("InfluxDB 모두에서 데이터를 로드하지 못했습니다.")
            return
            # 필요 시 예외 처리 또는 종료
        else:
            logger.info("데이터 로드 완료.")
            
        # 설정 최적화 (선택 사항)
        if config.get("advanced", {}).get("optimize_config", False):
            logger.info("데이터셋에 맞게 설정 최적화 중...")
            optimized_config = get_optimal_config_for_dataset(config, df)
            if optimized_config:
                config = optimized_config
                logger.info("설정이 최적화되었습니다.")
        
        # 자원 사용량 분석 실행
        success, results = run_resource_analysis(config, df)
        
        if success:
            logger.info("자원 사용량 분석이 성공적으로 완료되었습니다.")
            # 필요한 경우 여기에 추가 처리 작업 수행
        else:
            logger.warning("자원 사용량 분석이 실패했거나 결과가 없습니다.")
    
    except Exception as e:
        logger.error(f"자원 사용량 분석 중 오류 발생: {e}", exc_info=True)

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='LSTM 기반 IoT 예측 시스템')
    
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'failure', 'resource'],
                        help='실행 모드 (all, failure, resource)')
    
    parser.add_argument('--config', type=str, default='config.json',
                        help='설정 파일 경로')
    
    parser.add_argument('--test', action='store_true',
                        help='테스트 데이터 생성 후 실행')
    
    parser.add_argument('--schedule', action='store_true',
                        help='스케줄러 실행 (정기적 예측)')
    
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV 파일 경로. "all"을 지정하면 모든 데이터 파일 로드')
    
    parser.add_argument('--visualize', action='store_true',
                        help='데이터 시각화 실행')
    
    parser.add_argument('--days', type=int, default=30,
                        help='테스트 데이터 생성 기간(일)')
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    global logger
    logger = setup_logging(config)
    
    logger.info("====== IoT 예측 시스템 시작 ======")
    logger.info(f"실행 모드: {args.mode}")
    
    # 필요한 디렉토리 생성
    ensure_dir(config["results"]["save_dir"])
    ensure_dir(config["results"].get("model_dir", "model_weights"))
    ensure_dir(os.path.dirname(config["logging"]["log_file"]))
    
    # 데이터 로드 변수
    df = None
    
    # 테스트 모드 설정 업데이트
    if args.test:
        config["test"]["enabled"] = True
        config["test"]["days"] = args.days
    
    # 테스트 데이터 생성 옵션이 있는 경우
    if config["test"]["enabled"]:
        logger.info(f"테스트 데이터 생성 중 ({config['test']['days']}일)...")
        df = generate_test_data(
            days=config["test"]["days"], 
            hour_interval=config["test"]["hour_interval"]
        )
        
        if df is not None and not df.empty:
            logger.info(f"테스트 데이터 생성 완료: {len(df)}행, {df.shape[1]}열")
        else:
            logger.error("테스트 데이터 생성 실패")
            return
    
    # CSV 파일 로드 옵션이 있는 경우
    elif args.csv:
        if args.csv == 'all':  # 모든 데이터 파일 로드
            logger.info("모든 데이터 파일 로드 중...")
            df = load_all_data()
        else:  # 특정 CSV 파일 로드
            logger.info(f"CSV 파일 로드 중: {args.csv}")
            from utils.data_loader import load_influxdb_csv_data
            df = load_influxdb_csv_data(args.csv)
        
        if df is None or df.empty:
            logger.error("CSV 파일을 로드할 수 없습니다.")
            return
        
        logger.info(f"CSV 데이터 로드 완료: {len(df)}행, {df.shape[1]}열")
    
    # 스케줄러 실행 모드
    if args.schedule:
        logger.info("스케줄러 실행 모드: 정기적 예측 시작")
        
        # 데이터베이스 연결 테스트 (테스트 모드가 아닐 경우만)
        if not args.test and not args.csv:
            if 'influxdb' in config:
                logger.info("InfluxDB 연결 테스트 중...")
                influxdb_ok = test_influxdb_connection(config)
                if not influxdb_ok:
                    logger.error("InfluxDB 연결에 실패했습니다. 설정을 확인하세요.")
                    return
            
            if 'mysql' in config:
                logger.info("MySQL 연결 테스트 중...")
                mysql_ok = test_mysql_connection(config)
                if not mysql_ok:
                    logger.warning("MySQL 연결에 실패했습니다. 설정을 확인하세요.")
                    logger.info("MySQL 저장 기능이 비활성화됩니다.")
                    config.pop('mysql', None)
                else:
                    # MySQL 초기화
                    init_mysql(config)
        
        # 고장 예측 스케줄링
        if args.mode in ['all', 'failure'] and config["failure_prediction"]["enabled"]:
            interval = config["failure_prediction"]["interval_hours"]
            schedule.every(interval).hours.do(scheduled_failure_prediction, config)
            logger.info(f"고장 예측: {interval}시간마다 실행 예약됨")
        
        # 자원 사용량 분석 스케줄링
        if args.mode in ['all', 'resource'] and config["resource_analysis"]["enabled"]:
            interval = config["resource_analysis"]["interval_hours"]
            schedule.every(interval).hours.do(scheduled_resource_analysis, config)
            logger.info(f"자원 사용량 분석: {interval}시간마다 실행 예약됨")
        
        # 초기 실행
        logger.info("초기 예측 실행 중...")
        
        if args.mode in ['all', 'failure'] and config["failure_prediction"]["enabled"]:
            logger.info("고장 예측 초기 실행 중...")
            scheduled_failure_prediction(config)
            
        if args.mode in ['all', 'resource'] and config["resource_analysis"]["enabled"]:
            logger.info("자원 사용량 분석 초기 실행 중...")
            scheduled_resource_analysis(config)
        
        # 스케줄러 유지
        try:
            logger.info("스케줄러 실행 중...")
            logger.info("종료하려면 Ctrl+C를 누르세요.")
            
            # 다음 작업 시간 표시
            next_runs = []
            for job in schedule.jobs:
                next_run = job.next_run
                if next_run:
                    time_diff = next_run - datetime.now()
                    next_runs.append(f"{job.job_func.__name__}: {time_diff.total_seconds() // 60}분 후")
            
            if next_runs:
                logger.info(f"다음 예정된 작업: {', '.join(next_runs)}")
                
            # 스케줄러 실행 루프
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("사용자에 의해 스케줄러가 중지되었습니다.")
        
        except Exception as e:
            logger.error(f"스케줄러 실행 중 오류 발생: {e}", exc_info=True)
    
    # 일회성 실행 모드
    else:
        logger.info("일회성 실행 모드")
        
        # 데이터가 로드되지 않은 경우 로드
        if df is None:
            logger.info("데이터 로드 중...")
            df = load_all_data()
            
            if df is None or df.empty:
                logger.error("데이터를 로드할 수 없습니다. 종료합니다.")
                return
            
            logger.info(f"데이터 로드 완료: {len(df)}행, {df.shape[1]}열")
        
        # 시각화 옵션이 있는 경우
        if args.visualize:
            logger.info("데이터 시각화 실행 중...")
            from utils.visualization import plot_sensor_data, plot_correlation_matrix
            
            viz_dir = os.path.join(config["results"]["save_dir"], "visualizations")
            ensure_dir(viz_dir)
            
            # 주요 측정 항목 시각화
            important_cols = []
            for category in ['cpu', 'memory', 'disk', 'sensors']:
                cols = [col for col in df.columns if any(kw in col.lower() for kw in category.split('_'))]
                if cols:
                    important_cols.extend(cols[:2])  # 카테고리당 최대 2개 열만 선택
            
            if important_cols:
                plot_sensor_data(
                    df, 
                    cols=important_cols,
                    save_path=os.path.join(viz_dir, "key_metrics.png")
                )
                logger.info(f"주요 측정 항목 시각화 저장: {os.path.join(viz_dir, 'key_metrics.png')}")
            
            # 상관관계 행렬 시각화
            plot_correlation_matrix(
                df,
                save_path=os.path.join(viz_dir, "correlation_matrix.png")
            )
            logger.info(f"상관관계 행렬 시각화 저장: {os.path.join(viz_dir, 'correlation_matrix.png')}")
        
        # 고장 예측 실행
        if args.mode in ['all', 'failure'] and config["failure_prediction"]["enabled"]:
            logger.info("고장 예측 실행 중...")
            success, results = run_failure_prediction(config, df)
            
            if success:
                logger.info("고장 예측이 성공적으로 완료되었습니다.")
            else:
                logger.warning("고장 예측이 실패했거나 결과가 없습니다.")
        
        # 자원 사용량 분석 실행
        if args.mode in ['all', 'resource'] and config["resource_analysis"]["enabled"]:
            logger.info("자원 사용량 분석 실행 중...")
            try:
                success, results = run_resource_analysis(config, df)
            except TypeError:
                logger.error("자원 사용량 분석 함수가 예상된 반환값을 제공하지 않았습니다.")
                success, results = False, None
        
            if success:
                logger.info("자원 사용량 분석이 성공적으로 완료되었습니다.")
            else:
                logger.warning("자원 사용량 분석이 실패했거나 결과가 없습니다.")
    
    logger.info("====== IoT 예측 시스템 실행 완료 ======")

if __name__ == "__main__":
    main()