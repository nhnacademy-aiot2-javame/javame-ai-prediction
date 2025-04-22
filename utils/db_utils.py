#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터베이스 연결 및 쿼리 유틸리티

MySQL, InfluxDB 연결 및 쿼리를 위한 함수 제공
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

# 로깅 설정
logger = logging.getLogger(__name__)

def test_mysql_connection(config):
    """
    MySQL 연결 테스트
    
    Args:
        config (dict): MySQL 설정
        
    Returns:
        bool: 연결 성공 여부
    """
    logger.info("=== MySQL 연결 테스트 ===")
    
    mysql_config = config.get("mysql", {})
    
    if not all(key in mysql_config for key in ["host", "user", "password"]):
        logger.error("MySQL 설정이 불완전합니다. host, user, password가 필요합니다.")
        return False
    
    try:
        # 포트 정보 확인
        port = mysql_config.get("port", 3306)
        
        logger.info(f"MySQL 연결 시도: {mysql_config['host']}:{port}")
        
        # 데이터베이스 직접 연결 시도
        conn = mysql.connector.connect(
            host=mysql_config["host"],
            port=port,
            user=mysql_config["user"],
            password=mysql_config["password"],
            database=mysql_config["database"]
        )
        
        logger.info(f"데이터베이스 '{mysql_config['database']}' 연결 성공!")
        
        # 테이블 목록 확인
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        if tables:
            table_names = [table[0] for table in tables]
            logger.info(f"데이터베이스에 {len(table_names)}개 테이블이 있습니다: {table_names}")
        else:
            logger.info("데이터베이스에 테이블이 없습니다.")
            # 테이블 없으면 필요한 테이블 생성
            init_mysql(config)
        
        cursor.close()
        logger.info("MySQL 연결 테스트 성공!")
        return True
        
    except Error as e:
        logger.error(f"MySQL 연결 실패: {e}")
        return False
    
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

def test_influxdb_connection(config):
    """
    InfluxDB 연결 테스트
    
    Args:
        config (dict): InfluxDB 설정
        
    Returns:
        bool: 연결 성공 여부
    """
    logger.info("=== InfluxDB 연결 테스트 ===")
    
    influx_config = config.get("influxdb", {})
    
    if not all(key in influx_config for key in ["url", "token", "org", "bucket"]):
        logger.error("InfluxDB 설정이 불완전합니다. url, token, org, bucket이 필요합니다.")
        return False
    
    try:
        logger.info(f"InfluxDB 연결 시도: {influx_config['url']}")
        
        # 클라이언트 생성
        client = InfluxDBClient(
            url=influx_config["url"],
            token=influx_config["token"],
            org=influx_config["org"]
        )
        
        # 간단한 쿼리로 연결 테스트
        query_api = client.query_api()
        
        # 버킷 존재 확인
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets()
        bucket_names = [bucket.name for bucket in buckets.buckets]
        
        if influx_config["bucket"] in bucket_names:
            logger.info(f"버킷 '{influx_config['bucket']}' 확인됨.")
        else:
            logger.warning(f"버킷 '{influx_config['bucket']}'를 찾을 수 없습니다.")
            logger.info(f"사용 가능한 버킷: {bucket_names}")
            return False
        
        # 샘플 쿼리 실행
        measurement = influx_config.get("measurement", "")
        if measurement:
            try:
                # 지난 24시간 데이터 쿼리
                start_time = datetime.now() - pd.Timedelta(hours=24)
                query = f'''
                from(bucket: "{influx_config["bucket"]}")
                |> range(start: {start_time.strftime("%Y-%m-%dT%H:%M:%SZ")})
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> limit(n: 5)
                '''
                
                tables = query_api.query(query=query)
                
                if tables:
                    record_count = sum(1 for table in tables for _ in table.records)
                    logger.info(f"쿼리 성공: {record_count}개 레코드 검색됨")
                else:
                    logger.warning(f"쿼리 결과가 없습니다. measurement '{measurement}'를 확인하세요.")
            except Exception as e:
                logger.warning(f"샘플 쿼리 실행 중 오류: {e}")
        
        logger.info("InfluxDB 연결 테스트 성공!")
        return True
        
    except Exception as e:
        logger.error(f"InfluxDB 연결 실패: {e}")
        return False
    
    finally:
        if 'client' in locals():
            client.close()

def init_mysql(config):
    """
    MySQL 데이터베이스 초기화 - 테이블 생성
    
    Args:
        config (dict): MySQL 연결 설정
        
    Returns:
        bool: 초기화 성공 여부
    """
    mysql_config = config["mysql"]
    conn = None
    
    try:
        # MySQL 연결
        logger.info(f"MySQL 테이블 초기화 연결 시도: {mysql_config['host']}:{mysql_config.get('port', 3306)}")
        conn = mysql.connector.connect(
            host=mysql_config["host"],
            port=mysql_config.get("port", 3306),
            user=mysql_config["user"],
            password=mysql_config["password"],
            database=mysql_config["database"]
        )
        
        cursor = conn.cursor()
        
        # 테이블 생성
        # 1. 고장 예측 결과 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS failure_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME,
            device_id VARCHAR(50),
            failure_probability FLOAT,
            is_failure_predicted BOOLEAN,
            prediction_time DATETIME,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 2. 자원 사용량 예측 결과 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS resource_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME,
            device_id VARCHAR(50),
            resource_type VARCHAR(50),
            predicted_value FLOAT,
            prediction_time DATETIME,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 3. 용량 증설 계획 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS capacity_planning (
            id INT AUTO_INCREMENT PRIMARY KEY,
            scenario VARCHAR(50),
            resource_type VARCHAR(50),
            expansion_date DATE,
            threshold_value FLOAT,
            prediction_time DATETIME,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 4. 예측 실행 로그 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_runs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            run_mode VARCHAR(20),
            status VARCHAR(20),
            start_time DATETIME,
            end_time DATETIME,
            execution_time_sec INT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        logger.info("MySQL 데이터베이스 테이블 초기화 완료")
        return True
        
    except Error as e:
        logger.error(f"MySQL 초기화 중 오류 발생: {e}")
        return False
        
    finally:
        if conn is not None and conn.is_connected():
            cursor.close()
            conn.close()

def save_to_mysql(df, table_name, config):
    """
    데이터프레임을 MySQL에 저장
    
    Args:
        df (pd.DataFrame): 저장할 데이터프레임
        table_name (str): 테이블 이름
        config (dict): MySQL 설정
        
    Returns:
        bool: 성공 여부
    """
    mysql_config = config["mysql"]
    conn = None
    
    try:
        conn = mysql.connector.connect(
            host=mysql_config["host"],
            port=mysql_config.get("port", 3306),
            user=mysql_config["user"],
            password=mysql_config["password"],
            database=mysql_config["database"]
        )
        
        cursor = conn.cursor()
        
        # 컬럼 목록 생성
        columns_str = ', '.join([f"`{col}`" for col in df.columns])
        placeholders = ', '.join(['%s'] * len(df.columns))
        
        # 삽입 쿼리
        insert_query = f"""
        INSERT INTO {table_name} ({columns_str})
        VALUES ({placeholders})
        """
        
        # 데이터 변환
        values = []
        for _, row in df.iterrows():
            row_values = []
            for val in row:
                if isinstance(val, (np.int64, np.int32)):
                    val = int(val)
                elif isinstance(val, (np.float64, np.float32)):
                    val = float(val)
                row_values.append(val)
            values.append(tuple(row_values))
        
        # 일괄 삽입
        cursor.executemany(insert_query, values)
        conn.commit()
        
        logger.info(f"{len(df)}개 행이 테이블 '{table_name}'에 저장되었습니다.")
        return True
        
    except Error as e:
        logger.error(f"MySQL 저장 중 오류 발생: {e}")
        return False
        
    finally:
        if conn is not None and conn.is_connected():
            cursor.close()
            conn.close()

def store_to_influxdb(df, config, measurement=None, tags=None):
    """
    데이터프레임을 InfluxDB에 저장
    
    Args:
        df (pd.DataFrame): 저장할 데이터프레임 (인덱스는 타임스탬프여야 함)
        config (dict): InfluxDB 설정
        measurement (str): 측정 이름 (None이면 설정의 measurement 사용)
        tags (dict): 태그 정보 딕셔너리
        
    Returns:
        bool: 성공 여부
    """
    influx_config = config["influxdb"]
    measurement = measurement or influx_config["measurement"]
    
    if df.empty:
        logger.warning("저장할 데이터가 비어 있습니다.")
        return False
    
    # 인덱스가 타임스탬프인지 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("InfluxDB 저장 오류: 데이터프레임 인덱스가 DatetimeIndex가 아닙니다.")
        return False
    
    try:
        client = InfluxDBClient(
            url=influx_config["url"],
            token=influx_config["token"],
            org=influx_config["org"]
        )
        
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # 기본 태그
        if tags is None:
            tags = {"device_id": "device_001"}
        
        # 데이터 포인트 작성
        records = []
        for timestamp, row in df.iterrows():
            for col in df.columns:
                # 숫자형 데이터만 저장
                if pd.api.types.is_numeric_dtype(row[col]):
                    records.append({
                        "measurement": measurement,
                        "tags": tags,
                        "fields": {col: float(row[col])},
                        "time": timestamp
                    })
        
        # 일괄 저장
        write_api.write(
            bucket=influx_config["bucket"],
            record=records
        )
        
        logger.info(f"{len(records)}개 데이터 포인트가 InfluxDB '{measurement}'에 저장되었습니다.")
        return True
        
    except Exception as e:
        logger.error(f"InfluxDB 저장 중 오류 발생: {e}")
        return False
        
    finally:
        if 'client' in locals():
            write_api.close()
            client.close()