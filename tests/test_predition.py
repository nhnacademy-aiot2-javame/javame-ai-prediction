#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IoT 예측 시스템 테스트 스크립트

유닛 테스트와 통합 테스트 제공
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 상위 디렉토리를 sys.path에 추가하여 모듈 임포트 가능하게 함
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 테스트할 모듈 임포트
from utils.env_loader import load_environment_variables, update_config_from_env
from utils.data_loader import generate_test_data, preprocess_data
from utils.batch_monitor import BatchMonitor

# 테스트 설정
TEST_CONFIG = {
    "influxdb": {
        "url": "http://localhost:8086",
        "token": "test_token",
        "org": "test_org",
        "bucket": "test_bucket",
        "measurement": "system"
    },
    "mysql": {
        "host": "localhost",
        "port": 3306,
        "user": "test_user",
        "password": "test_password",
        "database": "test_db"
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
        "interval_hours": 2,
        "input_window": 48,
        "target_column": "user",
        "capacity_threshold": 80
    },
    "results": {
        "save_dir": "test_results",
        "save_plots": False,
        "save_csv": True,
        "model_dir": "test_models",
        "log_dir": "test_logs"
    },
    "logging": {
        "level": "INFO",
        "log_file": "test_logs/test.log",
        "max_size_mb": 1,
        "backup_count": 1
    },
    "test": {
        "enabled": True,
        "days": 7,
        "hour_interval": 1
    }
}

# 테스트 픽스처
@pytest.fixture
def test_config():
    """테스트용 설정 제공"""
    return TEST_CONFIG.copy()

@pytest.fixture
def test_data():
    """테스트용 데이터 생성"""
    return generate_test_data(days=3, hour_interval=1)

# 환경 변수 로드 테스트
def test_load_environment_variables(monkeypatch):
    """환경 변수 로드 함수 테스트"""
    # 임시 환경 변수 설정
    monkeypatch.setenv("INFLUXDB_URL", "http://test-influxdb:8086")
    monkeypatch.setenv("MYSQL_HOST", "test-mysql")
    monkeypatch.setenv("PREDICTION_INTERVAL", "3")
    
    # 환경 변수 로드
    env_vars = load_environment_variables()
    
    # 검증
    assert env_vars["INFLUXDB_URL"] == "http://test-influxdb:8086"
    assert env_vars["MYSQL_HOST"] == "test-mysql"
    assert env_vars["PREDICTION_INTERVAL"] == 3

# 설정 파일 업데이트 테스트
def test_update_config_from_env(monkeypatch, tmpdir):
    """설정 파일 업데이트 테스트"""
    # 임시 환경 변수 설정
    monkeypatch.setenv("INFLUXDB_URL", "http://test-influxdb:8086")
    monkeypatch.setenv("MYSQL_HOST", "test-mysql")
    monkeypatch.setenv("PREDICTION_INTERVAL", "3")
    
    # 임시 설정 파일 경로
    config_path = tmpdir.join("test_config.json")
    
    # 기존 설정 파일 생성
    with open(config_path, 'w') as f:
        json.dump(TEST_CONFIG, f)
    
    # 설정 업데이트
    updated_config = update_config_from_env(config_path)
    
    # 검증
    assert updated_config["influxdb"]["url"] == "http://test-influxdb:8086"
    assert updated_config["mysql"]["host"] == "test-mysql"
    assert updated_config["failure_prediction"]["interval_hours"] == 3

# 데이터 생성 테스트
def test_generate_test_data():
    """테스트 데이터 생성 테스트"""
    # 테스트 데이터 생성
    days = 2
    hour_interval = 2
    df = generate_test_data(days=days, hour_interval=hour_interval)
    
    # 검증
    expected_rows = (days * 24) // hour_interval + 1
    assert len(df) == expected_rows
    assert isinstance(df.index, pd.DatetimeIndex)
    assert 'usage_user' in df.columns
    assert 'temp_input' in df.columns

# 데이터 전처리 테스트
def test_preprocess_data(test_data):
    """데이터 전처리 테스트"""
    # 결측치 추가
    test_data.iloc[5, 0] = np.nan
    
    # 전처리
    processed_df = preprocess_data(test_data, 'temp_input')
    
    # 검증
    assert processed_df.isnull().sum().sum() == 0  # 결측치 없음
    assert len(processed_df) == len(test_data)  # 데이터 길이 보존

# 배치 모니터링 클래스 테스트
def test_batch_monitor(monkeypatch):
    """배치 모니터링 클래스 테스트"""
    # MySQL 연결 모킹
    class MockConnection:
        def is_connected(self):
            return True
        
        def cursor(self):
            return MockCursor()
        
        def close(self):
            pass
        
        def commit(self):
            pass
    
    class MockCursor:
        def execute(self, query, params=None):
            return None
        
        def fetchone(self):
            return [datetime.now()]
        
        def fetchall(self):
            return []
        
        def close(self):
            pass
        
        @property
        def lastrowid(self):
            return 1
    
    # MySQL 커넥터 모킹
    def mock_connect(**kwargs):
        return MockConnection()
    
    monkeypatch.setattr('mysql.connector.connect', mock_connect)
    
    # 배치 모니터 테스트
    monitor = BatchMonitor(TEST_CONFIG)
    assert monitor.connect() == True
    
    # 실행 로그 테스트
    run_id = monitor.log_run_start('test')
    assert run_id == 1
    
    # 실행 종료 로그 테스트
    assert monitor.log_run_end(run_id, 'SUCCESS') == True

# 통합 테스트: 환경 변수에서 설정 로드 및 배치 모니터링
def test_integration_env_config_monitor(monkeypatch, tmpdir):
    """환경 변수, 설정 파일, 배치 모니터링 통합 테스트"""
    # 임시 환경 변수 설정
    monkeypatch.setenv("INFLUXDB_URL", "http://test-influxdb:8086")
    monkeypatch.setenv("MYSQL_HOST", "test-mysql")
    
    # 임시 설정 파일 경로
    config_path = tmpdir.join("test_integration.json")
    
    # 설정 업데이트
    updated_config = update_config_from_env(config_path)
    
    # MySQL 연결을 모킹하여 배치 모니터 테스트
    class MockConnection:
        def is_connected(self):
            return True
        
        def cursor(self):
            return MockCursor()
        
        def close(self):
            pass
        
        def commit(self):
            pass
    
    class MockCursor:
        def execute(self, query, params=None):
            return None
        
        def fetchone(self):
            return [datetime.now()]
        
        def fetchall(self):
            return []
        
        def close(self):
            pass
        
        @property
        def lastrowid(self):
            return 1
    
    def mock_connect(**kwargs):
        # 올바른 호스트로 연결 시도했는지 확인
        assert kwargs['host'] == 'test-mysql'
        return MockConnection()
    
    monkeypatch.setattr('mysql.connector.connect', mock_connect)
    
    # 배치 모니터 테스트
    monitor = BatchMonitor(updated_config)
    assert monitor.connect() == True

if __name__ == "__main__":
    pytest.main()