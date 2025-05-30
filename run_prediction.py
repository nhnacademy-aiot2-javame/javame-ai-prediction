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
from utils.api_sender import APISender, get_api_url

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
        logger.info("고장 예측을 위한 데이터 로드 중...")
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)  # 30일 데이터

        df = query_influxdb(config, start_time, origins=["server_data", "sensor_data"])

        if df is None or df.empty:
            logger.error("데이터를 로드할 수 없습니다. 고장 예측을 건너뜁니다.")
            return
        else:
            logger.info("데이터 로드 완료.")
            
        # 고장 예측 실행
        success, results = run_failure_prediction(config, df)
        
        if success:
            logger.info("고장 예측이 성공적으로 완료되었습니다.")
            
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
        start_time = end_time - timedelta(days=90)  # 90일 데이터 (장기 추세 분석용)

        df = query_influxdb(config, start_time, origins=["server_data", "sensor_data"])

        if df is None or df.empty:
            logger.error("데이터를 로드할 수 없습니다. 자원 사용량 분석을 건너뜁니다.")
            return
        else:
            logger.info("데이터 로드 완료.")
        
        # 자원 사용량 분석 실행
        success, results = run_resource_analysis(config, df)
        
        if success:
            logger.info("자원 사용량 분석이 성공적으로 완료되었습니다.")
            
            # 증설 시점 출력
            if isinstance(results, dict) and 'expansion_dates' in results:
                logger.info("==== 리소스 증설 예측 시점 ====")
                for scenario, date in results['expansion_dates'].items():
                    if date:
                        logger.info(f"- {scenario} 시나리오: {date.strftime('%Y-%m-%d')}에 증설 필요")
                    else:
                        logger.info(f"- {scenario} 시나리오: 예측 기간 내 증설 필요 없음")
                
                # API로 결과 전송
                try:
                    api_sender = APISender(get_api_url())
                    api_success = api_sender.send_resource_prediction(results['short_term'])
                    if api_success:
                        logger.info("API로 자원 예측 결과 전송 성공")
                    else:
                        logger.warning("API로 자원 예측 결과 전송 실패")
                except Exception as e:
                    logger.error(f"API 결과 전송 중 오류: {e}")
        else:
            logger.warning("자원 사용량 분석이 실패했거나 결과가 없습니다.")
    
    except Exception as e:
        logger.error(f"자원 사용량 분석 중 오류 발생: {e}", exc_info=True)
def run_multi_resource_prediction(config):
    """
    위치별, 자원별 예측 실행
    
    Args:
        config (dict): 설정 정보
        
    Returns:
        dict: 위치별, 자원별 예측 결과
    """
    logger.info("======= 위치별, 자원별 예측 실행 =======")
    
    # 결과 저장용 딕셔너리
    results = {}
    
    # 사용 가능한 위치 목록 조회
    locations = get_available_locations(config)
    logger.info(f"예측 수행할 위치 목록: {locations}")
    
    # 리소스 유형 목록
    resource_types = ["cpu", "memory", "disk", "network", "temperature"]
    if "resources" in config:
        resource_types = list(config["resources"].keys())
    
    # 모델 관리자 생성
    from models.lstm_model import ResourceModelManager
    model_manager = ResourceModelManager(config)
    
    # 각 위치별로 처리
    for location in locations:
        logger.info(f"위치 '{location}'에 대한 예측 처리 중...")
        location_results = {}
        
        # 데이터 로드 (30일 데이터)
        df = query_influxdb(config, datetime.now() - timedelta(days=30), location=location)
        
        if df is None or df.empty:
            logger.warning(f"위치 '{location}'에 대한 데이터가 없습니다.")
            continue
        
        # 각 자원별 모델 학습 및 예측
        for resource_type in resource_types:
            # 리소스 설정 로드
            resource_config = config.get("resources", {}).get(resource_type, {})
            target_columns = resource_config.get("target_columns", [])
            
            # 특정 자원 타입에 대한 타겟 열이 설정에 없는 경우
            if not target_columns:
                # 기본 타겟 열 적용
                if resource_type == "cpu":
                    target_columns = ["usage_user", "usage_system", "usage_idle"]
                elif resource_type == "memory":
                    target_columns = ["used_percent", "available_percent"]
                elif resource_type == "disk":
                    target_columns = ["used_percent"]
                elif resource_type == "network":
                    target_columns = ["bytes_sent", "bytes_recv"]
                elif resource_type == "temperature":
                    target_columns = ["temp_input"]
            
            # 해당 자원에 관련된 컬럼이 있는지 확인
            available_columns = [col for col in target_columns if col in df.columns]
            
            if not available_columns:
                logger.info(f"위치 '{location}'에 '{resource_type}' 자원 데이터가 없습니다.")
                continue
            
            # 대표 타겟 컬럼 선택
            target_column = available_columns[0]
            logger.info(f"위치 '{location}'의 '{resource_type}' 자원 타겟 컬럼: {target_column}")
            
            # 모델 생성 또는 가져오기
            model = model_manager.get_or_create_model(
                location, 
                resource_type, 
                df.shape[1], 
                resource_config.get("input_window", config["failure_prediction"]["input_window"])
            )
            
            # 데이터 전처리
            processed_df = preprocess_data(df, target_column)
            
            # 모델 학습
            logger.info(f"위치 '{location}'의 '{resource_type}' 자원에 대한 모델 학습 중...")
            try:
                if isinstance(model, FailureLSTM):
                    threshold = resource_config.get("failure_threshold", config["failure_prediction"]["threshold"])
                    model.fit(processed_df, target_column, failure_threshold=threshold)
                else:
                    model.fit(processed_df, target_column)
                
                # 예측 수행
                if isinstance(model, FailureLSTM):
                    pred_results = model.predict(processed_df)
                    location_results[resource_type] = {
                        "failure_prediction": pred_results
                    }
                else:
                    pred_results = model.predict(processed_df, target_column)
                    try:
                        capacity_threshold = resource_config.get(
                            "capacity_threshold", 
                            config["resource_analysis"]["capacity_threshold"]
                        )
                        scenario_results, expansion_dates = model.predict_long_term(
                            periods=180,
                            capacity_threshold=capacity_threshold
                        )
                        location_results[resource_type] = {
                            "resource_prediction": pred_results,
                            "long_term": scenario_results,
                            "expansion_dates": expansion_dates
                        }
                    except Exception as e:
                        logger.error(f"장기 예측 실패: {e}")
                        location_results[resource_type] = {
                            "resource_prediction": pred_results
                        }
                
                logger.info(f"위치 '{location}'의 '{resource_type}' 자원 예측 완료")
                
                # 결과 시각화 및 저장
                if config["results"]["save_plots"]:
                    save_dir = os.path.join(
                        config["results"]["save_dir"],
                        location,
                        resource_type,
                        datetime.now().strftime("%Y%m%d_%H%M%S")
                    )
                    
                    os.makedirs(save_dir, exist_ok=True)
                    
                    if isinstance(model, FailureLSTM):
                        plot_failure_prediction(
                            processed_df, 
                            target_column, 
                            pred_results,
                            threshold=threshold,
                            save_path=os.path.join(save_dir, "failure_prediction_results.png")
                        )
                    else:
                        plot_resource_prediction(
                            processed_df, 
                            target_column, 
                            pred_results,
                            save_path=os.path.join(save_dir, "resource_prediction_results.png")
                        )
                        
                        if "long_term" in location_results[resource_type]:
                            plot_long_term_scenarios(
                                location_results[resource_type]["long_term"],
                                capacity_threshold=capacity_threshold,
                                expansion_dates=location_results[resource_type]["expansion_dates"],
                                save_path=os.path.join(save_dir, "long_term_scenarios.png")
                            )
                
            except Exception as e:
                logger.error(f"위치 '{location}'의 '{resource_type}' 자원 예측 중 오류 발생: {e}")
                logger.error(traceback.format_exc())
        
        # 예측 결과 통합
        try:
            aggregated_results = model_manager.aggregate_predictions(location, location_results)
            
            # 위치별 결과 저장
            results[location] = {
                "resources": location_results,
                "aggregated": aggregated_results
            }
            
            # MySQL 저장
            if 'mysql' in config:
                if "failure_prediction" in aggregated_results and aggregated_results["failure_prediction"] is not None:
                    save_to_mysql(
                        aggregated_results["failure_prediction"].reset_index(), 
                        "failure_predictions", 
                        config
                    )
                
                if "resource_prediction" in aggregated_results and aggregated_results["resource_prediction"] is not None:
                    save_to_mysql(
                        aggregated_results["resource_prediction"].reset_index(), 
                        "resource_predictions", 
                        config
                    )
            
            # API 전송
            try:
                api_sender = APISender(get_api_url())
                api_sender.send_multi_location_predictions(location, aggregated_results)
            except Exception as e:
                logger.error(f"API 결과 전송 중 오류: {e}")
            
            logger.info(f"위치 '{location}' 예측 결과 통합 및 저장 완료")
        except Exception as e:
            logger.error(f"위치 '{location}' 결과 통합 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
    
    # 결과 반환
    return results
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
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 로깅 설정
    global logger
    logger = setup_logging(config)
    
    logger.info("====== IoT 예측 시스템 시작 ======")
    logger.info(f"실행 모드: {args.mode}")
    
    # 디렉토리 생성
    ensure_dir(config["results"]["save_dir"])
    ensure_dir(config["results"].get("model_dir", "model_weights"))
    ensure_dir(os.path.dirname(config["logging"]["log_file"]))
    
    # 데이터 로드 변수 
    df = None
    
    # 테스트 모드 설정
    if args.test:
        config["test"]["enabled"] = True
        logger.info("테스트 모드 활성화")
    
    # 테스트 데이터 생성
    if config["test"]["enabled"]:
        logger.info(f"테스트 데이터 생성 중 ({config['test']['days']}일)...")
        df = generate_test_data(
            days=config['test']['days'], 
            hour_interval=config['test']['hour_interval']
        )
        
        if df is None or df.empty:
            logger.error("테스트 데이터 생성 실패")
            return
        
        logger.info(f"테스트 데이터 생성 완료: {len(df)}행, {df.shape[1]}열")
    
    # CSV 파일 로드
    elif args.csv:
        # CSV 파일 처리 로직 (기존 코드와 동일)
        pass
    
    # 스케줄러 실행 모드
    if args.schedule:
        logger.info("스케줄러 실행 모드: 정기적 예측 시작")
        
        # 데이터베이스 연결 테스트 (테스트 모드가 아닐 경우만)
        if not args.test and not args.csv:
            # 데이터베이스 연결 테스트 로직 (기존 코드와 동일)
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
            
            # 다음 작업 시간 표시 (기존 코드와 동일)
            display_next_jobs()
            
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
            logger.info("InfluxDB에서 데이터 로드 중...")
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)  # 30일 데이터 로드
            
            # origin 태그 기반으로 데이터 로드
            df = query_influxdb(config, start_time, origins=["server_data", "sensor_data"])
            
            if df is None or df.empty:
                logger.error("데이터를 로드할 수 없습니다. 종료합니다.")
                return
            
            logger.info(f"데이터 로드 완료: {len(df)}행, {df.shape[1]}열")
        
        # 시각화 옵션
        if args.visualize:
            # 데이터 시각화 로직 (기존 코드와 유사)
            logger.info("데이터 시각화 실행 중...")
            # 서버-환경 상관관계 시각화 추가
            server_cols = ['usage_user', 'available_percent', 'load1']
            env_cols = ['sensor_data_temperature', 'sensor_data_humidity']
            
            viz_dir = os.path.join(config["results"]["save_dir"], "visualizations")
            ensure_dir(viz_dir)
            
            plot_server_environment_correlation(
                df, server_cols, env_cols,
                save_path=os.path.join(viz_dir, "server_environment_correlation.png")
            )
            logger.info("서버-환경 상관관계 시각화 완료")
        
        # 고장 예측 실행
        if args.mode in ['all', 'failure'] and config["failure_prediction"]["enabled"]:
            logger.info("고장 예측 실행 중...")
            failure_success, failure_results = run_failure_prediction(config, df)
            
            if failure_success:
                logger.info("고장 예측이 성공적으로 완료되었습니다.")
            else:
                logger.warning("고장 예측이 실패했거나 결과가 없습니다.")
        
        # 자원 사용량 분석 실행
        if args.mode in ['all', 'resource'] and config["resource_analysis"]["enabled"]:
            logger.info("자원 사용량 분석 실행 중...")
            resource_success, resource_results = run_resource_analysis(config, df)
            
            if resource_success:
                logger.info("자원 사용량 분석이 성공적으로 완료되었습니다.")
                
                # 증설 시점 출력
                if isinstance(resource_results, dict) and 'expansion_dates' in resource_results:
                    logger.info("==== 리소스 증설 예측 시점 ====")
                    for scenario, date in resource_results['expansion_dates'].items():
                        if date:
                            logger.info(f"- {scenario} 시나리오: {date.strftime('%Y-%m-%d')}에 증설 필요")
                        else:
                            logger.info(f"- {scenario} 시나리오: 예측 기간 내 증설 필요 없음")
            else:
                logger.warning("자원 사용량 분석이 실패했거나 결과가 없습니다.")
    
    logger.info("====== IoT 예측 시스템 실행 완료 ======")

if __name__ == "__main__":
    main()