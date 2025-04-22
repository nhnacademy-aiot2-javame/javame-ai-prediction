#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IoT 예측 시스템 배치 애플리케이션 메인 스크립트

단일 실행 또는 스케줄러 모드로 실행 가능
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# 사용자 정의 모듈 임포트
from utils.env_loader import update_config_from_env
from batch_runner import run_prediction_job, test_database_connections, setup_logging

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='IoT 예측 시스템 배치 애플리케이션')
    
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'failure', 'resource'],
                        help='실행 모드 (all, failure, resource)')
    
    parser.add_argument('--scheduler', action='store_true',
                        help='스케줄러 모드로 실행 (배치 러너 실행)')
    
    parser.add_argument('--test', action='store_true',
                        help='테스트 데이터로 실행')
    
    parser.add_argument('--check-db', action='store_true',
                        help='데이터베이스 연결 테스트')
    
    parser.add_argument('--config', type=str, default='config.json',
                        help='설정 파일 경로')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging()
    
    # 환경 변수 기반으로 설정 업데이트
    try:
        config = update_config_from_env(args.config)
        logger.info(f"설정 파일 업데이트 완료: {args.config}")
    except Exception as e:
        logger.error(f"설정 파일 업데이트 실패: {e}")
        sys.exit(1)
    
    # 테스트 모드 설정
    if args.test:
        logger.info("테스트 모드로 실행합니다.")
        config["test"]["enabled"] = True
        
        # 설정 저장
        with open(args.config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    
    # 데이터베이스 연결 테스트
    if args.check_db:
        logger.info("데이터베이스 연결 테스트를 실행합니다.")
        if test_database_connections():
            logger.info("데이터베이스 연결 테스트 성공.")
            sys.exit(0)
        else:
            logger.error("데이터베이스 연결 테스트 실패.")
            sys.exit(1)
    
    # 스케줄러 모드 또는 단일 실행 모드
    if args.scheduler:
        logger.info("스케줄러 모드로 실행합니다. (batch_runner.py)")
        os.system("python batch_runner.py")
    else:
        # 단일 실행 모드
        logger.info(f"단일 실행 모드로 실행합니다. (모드: {args.mode})")
        success = run_prediction_job(mode=args.mode)
        
        if success:
            logger.info(f"예측 작업 '{args.mode}' 모드 실행 성공.")
            sys.exit(0)
        else:
            logger.error(f"예측 작업 '{args.mode}' 모드 실행 실패.")
            sys.exit(1)

if __name__ == "__main__":
    main()