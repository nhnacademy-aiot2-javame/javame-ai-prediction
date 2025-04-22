#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
현재 데이터 구조에 맞게 config.json 파일을 업데이트하는 스크립트
"""

import json
import os
from datetime import datetime

def update_config():
    """config.json 파일을 현재 데이터 구조에 맞게 업데이트"""
    
    # 현재 config.json 로드
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # InfluxDB 설정 업데이트
    # 실제 measurement 중 하나를 선택 (여기서는 'system'을 기본으로 사용)
    config['influxdb']['measurement'] = 'system'
    
    # MySQL 설정 업데이트 (정확한 정보로)
    config['mysql']['host'] = 'localhost'
    config['mysql']['port'] = 3306
    config['mysql']['user'] = 'nhn_academy_221'
    config['mysql']['password'] = 'kK8ujeoz!'
    config['mysql']['database'] = 'nhn_academy_221'
    
    # 고장 예측 설정 업데이트
    # CPU 온도를 고장 징후로 사용
    config['failure_prediction']['target_column'] = 'temp_input'
    config['failure_prediction']['input_window'] = 24  # 24시간 데이터 참조
    config['failure_prediction']['threshold'] = 0.75  # 정규화된 값 기준
    
    # 자원 사용량 분석 설정 업데이트
    # CPU 사용률(usage_user)을 자원 사용량 지표로 사용
    config['resource_analysis']['target_column'] = 'user'
    config['resource_analysis']['input_window'] = 48  # 48시간 데이터 참조
    config['resource_analysis']['capacity_threshold'] = 80  # CPU 사용률 80% 이상을 위험수준으로 설정
    
    # 테스트 설정 업데이트
    config['test']['enabled'] = False  # 실제 데이터가 있으므로 테스트 비활성화
    
    # 로깅 레벨 디버그로 변경 (일시적)
    config['logging']['level'] = 'DEBUG'
    
    # 백업 파일 생성
    backup_path = f'config.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(backup_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"기존 설정 백업 완료: {backup_path}")
    
    # 업데이트된 설정 저장
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"설정 업데이트 완료: {config_path}")
    
    # 업데이트된 설정 출력
    print("\n업데이트된 주요 설정:")
    print(f"- InfluxDB measurement: {config['influxdb']['measurement']}")
    print(f"- 고장 예측 타겟: {config['failure_prediction']['target_column']}")
    print(f"- 자원 분석 타겟: {config['resource_analysis']['target_column']}")
    print(f"- MySQL 호스트: {config['mysql']['host']}")

if __name__ == "__main__":
    update_config()