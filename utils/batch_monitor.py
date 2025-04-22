#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
배치 작업 모니터링 유틸리티

배치 작업 실행 로그 기록 및 모니터링 기능 제공
MySQL 테이블에 실행 기록 저장 및 통계 수집
"""

import os
import time
import logging
import traceback
from datetime import datetime
import mysql.connector
from mysql.connector import Error
from functools import wraps

# 로깅 설정
logger = logging.getLogger(__name__)

class BatchMonitor:
    """배치 작업 모니터링 클래스"""
    
    def __init__(self, config):
        """
        초기화
        
        Args:
            config (dict): MySQL 연결 설정
        """
        self.config = config.get("mysql", {})
        self.conn = None
        
    def connect(self):
        """MySQL 연결 수립"""
        try:
            self.conn = mysql.connector.connect(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 3306),
                user=self.config.get("user", "root"),
                password=self.config.get("password", ""),
                database=self.config.get("database", "predictions")
            )
            return True
        except Error as e:
            logger.error(f"MySQL 연결 실패: {e}")
            return False
    
    def disconnect(self):
        """MySQL 연결 종료"""
        if self.conn and self.conn.is_connected():
            self.conn.close()
    
    def log_run_start(self, run_mode):
        """
        실행 시작 로그 기록
        
        Args:
            run_mode (str): 실행 모드 (failure, resource, all)
            
        Returns:
            int: 실행 ID (로그 레코드 ID)
        """
        if not self.conn or not self.conn.is_connected():
            if not self.connect():
                return None
        
        try:
            cursor = self.conn.cursor()
            
            # 로그 레코드 삽입
            query = """
            INSERT INTO prediction_runs 
            (run_mode, status, start_time) 
            VALUES (%s, %s, %s)
            """
            
            now = datetime.now()
            cursor.execute(query, (run_mode, 'RUNNING', now))
            
            self.conn.commit()
            run_id = cursor.lastrowid
            
            cursor.close()
            return run_id
            
        except Error as e:
            logger.error(f"실행 시작 로그 기록 실패: {e}")
            return None
    
    def log_run_end(self, run_id, status='SUCCESS', error_message=None):
        """
        실행 종료 로그 기록
        
        Args:
            run_id (int): 실행 ID
            status (str): 상태 (SUCCESS, FAILED)
            error_message (str): 오류 메시지
            
        Returns:
            bool: 성공 여부
        """
        if not run_id:
            return False
            
        if not self.conn or not self.conn.is_connected():
            if not self.connect():
                return False
        
        try:
            cursor = self.conn.cursor()
            
            # 시작 시간 조회
            cursor.execute("SELECT start_time FROM prediction_runs WHERE id = %s", (run_id,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"실행 ID {run_id}에 대한 레코드를 찾을 수 없습니다.")
                return False
                
            start_time = result[0]
            end_time = datetime.now()
            
            # 실행 시간 계산 (초)
            execution_time = int((end_time - start_time).total_seconds())
            
            # 로그 레코드 업데이트
            query = """
            UPDATE prediction_runs 
            SET status = %s, end_time = %s, execution_time_sec = %s, error_message = %s
            WHERE id = %s
            """
            
            cursor.execute(query, (status, end_time, execution_time, error_message, run_id))
            
            self.conn.commit()
            cursor.close()
            return True
            
        except Error as e:
            logger.error(f"실행 종료 로그 기록 실패: {e}")
            return False
    
    def log_prediction_counts(self, run_id, failure_count=0, resource_count=0):
        """
        예측 결과 개수 로깅
        
        Args:
            run_id (int): 실행 ID
            failure_count (int): 고장 예측 결과 개수
            resource_count (int): 자원 예측 결과 개수
            
        Returns:
            bool: 성공 여부
        """
        if not run_id:
            return False
            
        if not self.conn or not self.conn.is_connected():
            if not self.connect():
                return False
        
        try:
            cursor = self.conn.cursor()
            
            # 로그 레코드 업데이트
            query = """
            UPDATE prediction_runs 
            SET failure_count = %s, resource_count = %s
            WHERE id = %s
            """
            
            cursor.execute(query, (failure_count, resource_count, run_id))
            
            self.conn.commit()
            cursor.close()
            return True
            
        except Error as e:
            logger.error(f"예측 결과 개수 로깅 실패: {e}")
            return False
    
    def get_run_statistics(self, days=7):
        """
        실행 통계 조회
        
        Args:
            days (int): 조회 기간 (일)
            
        Returns:
            dict: 통계 정보
        """
        if not self.conn or not self.conn.is_connected():
            if not self.connect():
                return None
        
        try:
            cursor = self.conn.cursor(dictionary=True)
            
            # 성공/실패 개수
            query = """
            SELECT 
                run_mode,
                COUNT(*) AS total_runs,
                SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) AS success_count,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS failed_count,
                AVG(execution_time_sec) AS avg_execution_time
            FROM prediction_runs
            WHERE start_time >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY run_mode
            """
            
            cursor.execute(query, (days,))
            stats_by_mode = cursor.fetchall()
            
            # 일자별 실행 개수
            query = """
            SELECT 
                DATE(start_time) AS run_date,
                COUNT(*) AS total_runs,
                SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) AS success_count
            FROM prediction_runs
            WHERE start_time >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY DATE(start_time)
            ORDER BY run_date
            """
            
            cursor.execute(query, (days,))
            daily_stats = cursor.fetchall()
            
            cursor.close()
            
            return {
                'stats_by_mode': stats_by_mode,
                'daily_stats': daily_stats
            }
            
        except Error as e:
            logger.error(f"실행 통계 조회 실패: {e}")
            return None

def monitor_batch_job(config):
    """
    배치 작업 모니터링 데코레이터 함수
    
    Args:
        config (dict): 설정 정보
        
    Returns:
        callable: 데코레이터 함수
    """
    def decorator(func):
        @wraps(func)
        def wrapper(mode='all', *args, **kwargs):
            # 모니터 초기화
            monitor = BatchMonitor(config)
            run_id = monitor.log_run_start(mode)
            
            start_time = time.time()
            result = None
            error_message = None
            
            try:
                # 원래 함수 실행
                result = func(mode, *args, **kwargs)
                status = 'SUCCESS' if result else 'FAILED'
            except Exception as e:
                logger.error(f"배치 작업 실행 중 오류 발생: {e}")
                logger.error(traceback.format_exc())
                status = 'FAILED'
                error_message = str(e)
                result = False
            finally:
                # 실행 종료 로그 기록
                monitor.log_run_end(run_id, status, error_message)
                monitor.disconnect()
                
                execution_time = time.time() - start_time
                logger.info(f"배치 작업 실행 시간: {execution_time:.2f}초")
            
            return result
        
        return wrapper
    
    return decorator

if __name__ == "__main__":
    # 직접 실행 시 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 설정 로드
    import json
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # 테스트 실행
    monitor = BatchMonitor(config)
    
    # 연결 테스트
    if monitor.connect():
        logger.info("MySQL 연결 성공")
        
        # 통계 조회
        stats = monitor.get_run_statistics()
        if stats:
            logger.info(f"실행 통계: {stats}")
        
        monitor.disconnect()
    else:
        logger.error("MySQL 연결 실패")