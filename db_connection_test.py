#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InfluxDB와 MySQL 연결 확인 스크립트
config.json 파일을 이용해 연결 상태를 확인합니다.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta

# 데이터베이스 라이브러리
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import mysql.connector
from mysql.connector import Error

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("db_connection_tester")

def load_config(config_path):
    """설정 파일 로드"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        return None

def test_influxdb_connection(config):
    """InfluxDB 연결 테스트"""
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
        client.buckets_api().find_buckets()
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
        origins = influx_config.get("origins", ["server_data", "sensor_data"])
        
        if origins:
            try:
                # 지난 24시간 데이터 쿼리 - 첫 번째 origin으로 테스트
                start_time = datetime.now() - timedelta(hours=24)
                query = f'''
                from(bucket: "{influx_config["bucket"]}")
                |> range(start: {start_time.strftime("%Y-%m-%dT%H:%M:%SZ")})
                |> filter(fn: (r) => r["origin"] == "{origins[0]}")
                |> limit(n: 5)
                '''
                
                tables = query_api.query(query=query)
                
                if tables:
                    record_count = sum(1 for table in tables for _ in table.records)
                    logger.info(f"쿼리 성공: {record_count}개 레코드 검색됨")
                else:
                    logger.warning(f"쿼리 결과가 없습니다. origin '{origins[0]}'를 확인하세요.")
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

def test_mysql_connection(config):
    """MySQL 연결 테스트"""
    logger.info("=== MySQL 연결 테스트 ===")
    
    mysql_config = config.get("mysql", {})
    
    if not all(key in mysql_config for key in ["host", "user", "password"]):
        logger.error("MySQL 설정이 불완전합니다. host, user, password가 필요합니다.")
        return False
    
    try:
        # 포트 정보 확인
        port = mysql_config.get("port", 3306)
        
        logger.info(f"MySQL 연결 시도: {mysql_config['host']}:{port}")
        
        # 먼저 데이터베이스 없이 연결 시도
        conn = mysql.connector.connect(
            host=mysql_config["host"],
            port=port,
            user=mysql_config["user"],
            password=mysql_config["password"]
        )
        
        logger.info("기본 연결 성공!")
        
        # 데이터베이스 연결 시도
        database = mysql_config.get("database")
        if database:
            try:
                # 데이터베이스 연결 닫기
                conn.close()
                
                # 데이터베이스로 다시 연결
                conn = mysql.connector.connect(
                    host=mysql_config["host"],
                    port=port,
                    user=mysql_config["user"],
                    password=mysql_config["password"],
                    database=database
                )
                
                logger.info(f"데이터베이스 '{database}' 연결 성공!")
                
                # 테이블 목록 확인
                cursor = conn.cursor()
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                
                if tables:
                    table_names = [table[0] for table in tables]
                    logger.info(f"데이터베이스에 {len(table_names)}개 테이블이 있습니다: {table_names}")
                else:
                    logger.info("데이터베이스에 테이블이 없습니다.")
                
                cursor.close()
                
            except Error as e:
                logger.warning(f"데이터베이스 '{database}' 연결 실패: {e}")
                logger.info("데이터베이스가 존재하지 않거나 접근 권한이 없을 수 있습니다.")
                
                # 데이터베이스 생성 가능 여부 확인
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
                    logger.info(f"데이터베이스 '{database}' 생성 성공!")
                    cursor.close()
                except Error as e:
                    logger.error(f"데이터베이스 생성 실패: {e}")
        
        logger.info("MySQL 연결 테스트 성공!")
        return True
        
    except Error as e:
        logger.error(f"MySQL 연결 실패: {e}")
        return False
    
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.json"  # 기본값
    
    logger.info(f"설정 파일 '{config_path}'을(를) 사용합니다.")
    
    # 설정 로드
    config = load_config(config_path)
    if not config:
        return
    
    # 연결 테스트
    influxdb_ok = test_influxdb_connection(config)
    mysql_ok = test_mysql_connection(config)
    
    # 결과 요약
    logger.info("=== 연결 테스트 결과 ===")
    logger.info(f"InfluxDB 연결: {'성공' if influxdb_ok else '실패'}")
    logger.info(f"MySQL 연결: {'성공' if mysql_ok else '실패'}")
    
    if influxdb_ok and mysql_ok:
        logger.info("모든 연결이 정상입니다!")
    else:
        logger.warning("일부 연결에 문제가 있습니다. 설정을 확인하세요.")

if __name__ == "__main__":
    main()