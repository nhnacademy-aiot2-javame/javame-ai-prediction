#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API 결과 전송 유틸리티

예측 결과를 Spring Boot API로 전송하는 기능
"""

import json
import logging
import requests
from datetime import datetime
import pandas as pd

# 로깅 설정
logger = logging.getLogger(__name__)

class APISender:
    """예측 결과를 API로 전송하는 클래스"""
    
    def __init__(self, api_url="http://localhost:10272/ai/data"):
        """
        초기화
        
        Args:
            api_url (str): API 엔드포인트 URL
        """
        self.api_url = api_url
    
    def send_failure_prediction(self, results, location=None, resource_type=None):
        """
        고장 예측 결과 전송
        
        Args:
            results (pd.DataFrame): 고장 예측 결과
            location (str, optional): 위치 정보
            resource_type (str, optional): 자원 유형
                
        Returns:
            bool: 전송 성공 여부
        """
        if results is None or results.empty:
            logger.warning("전송할 고장 예측 결과가 없습니다.")
            return False
        
        try:
            # 데이터 변환
            payload = self._convert_failure_results(results, location, resource_type)
            
            # API 호출
            logger.info(f"고장 예측 결과 전송 중: {len(results)}개 레코드")
            response = requests.post(
                f"{self.api_url}/failure",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # 응답 확인
            if response.status_code == 200:
                logger.info(f"고장 예측 결과 전송 성공: {response.status_code}")
                logger.debug(f"API 응답: {response.text}")
                return True
            else:
                logger.error(f"고장 예측 결과 전송 실패: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"고장 예측 결과 전송 중 오류 발생: {e}")
            return False

    
    def send_resource_prediction(self, results, location=None, resource_type=None):
        """
        자원 사용량 예측 결과 전송
        
        Args:
            results (pd.DataFrame): 자원 사용량 예측 결과
            location (str, optional): 위치 정보
            resource_type (str, optional): 자원 유형
                
        Returns:
            bool: 전송 성공 여부
        """
        if results is None or results.empty:
            logger.warning("전송할 자원 사용량 예측 결과가 없습니다.")
            return False
        
        try:
            # 데이터 변환
            payload = self._convert_resource_results(results, location, resource_type)
            
            # API 호출
            logger.info(f"자원 사용량 예측 결과 전송 중: {len(results)}개 레코드")
            response = requests.post(
                f"{self.api_url}/resource",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # 응답 확인
            if response.status_code == 200:
                logger.info(f"자원 사용량 예측 결과 전송 성공: {response.status_code}")
                logger.debug(f"API 응답: {response.text}")
                return True
            else:
                logger.error(f"자원 사용량 예측 결과 전송 실패: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"자원 사용량 예측 결과 전송 중 오류 발생: {e}")
            return False
    
    def _convert_failure_results(self, results, location=None, resource_type=None):
        """
        고장 예측 결과를 API 요청 형식으로 변환
        
        Args:
            results (pd.DataFrame): 고장 예측 결과
            location (str, optional): 위치 정보
            resource_type (str, optional): 자원 유형
                
        Returns:
            dict: API 요청 페이로드
        """
        # 결과 복사 및 인덱스 리셋
        df = results.reset_index() if isinstance(results.index, pd.DatetimeIndex) else results.copy()
        
        # 타임스탬프를 ISO 형식 문자열로 변환
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        if 'prediction_time' in df.columns:
            df['prediction_time'] = df['prediction_time'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # location 및 resource_type 추가 (파라미터로 제공된 경우)
        if location is not None and 'location' not in df.columns:
            df['location'] = location
        
        if resource_type is not None and 'resource_type' not in df.columns:
            df['resource_type'] = resource_type
        
        # 기본 payload 구조
        payload = {
            "predictions": df.to_dict(orient="records"),
            "meta": {
                "model": "failure_lstm",
                "version": "1.0",
                "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            }
        }
        
        return payload
    
    def _convert_resource_results(self, results, location=None, resource_type=None):
        """
        자원 사용량 예측 결과를 API 요청 형식으로 변환
        
        Args:
            results (pd.DataFrame): 자원 사용량 예측 결과
            location (str, optional): 위치 정보
            resource_type (str, optional): 자원 유형
                
        Returns:
            dict: API 요청 페이로드
        """
        # 결과 복사 및 인덱스 리셋
        df = results.reset_index() if isinstance(results.index, pd.DatetimeIndex) else results.copy()
        
        # 타임스탬프를 ISO 형식 문자열로 변환
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        if 'prediction_time' in df.columns:
            df['prediction_time'] = df['prediction_time'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # location 및 resource_type 추가 (파라미터로 제공된 경우)
        if location is not None and 'location' not in df.columns:
            df['location'] = location
        
        if resource_type is not None and 'resource_type' not in df.columns:
            df['resource_type'] = resource_type
        
        # 기본 payload 구조
        payload = {
            "predictions": df.to_dict(orient="records"),
            "meta": {
                "model": "resource_lstm",
                "version": "1.0",
                "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            }
        }
        
        return payload
    def send_multi_location_predictions(self, location, results):
        """
        위치별 통합 예측 결과 전송
        
        Args:
            location (str): 위치 정보
            results (dict): 통합 예측 결과
            
        Returns:
            bool: 전송 성공 여부
        """
        success = True
        
        # 고장 예측 결과 전송
        if "failure_prediction" in results and results["failure_prediction"] is not None:
            if not self.send_failure_prediction(results["failure_prediction"], location):
                success = False
        
        # 자원 사용량 예측 결과 전송
        if "resource_prediction" in results and results["resource_prediction"] is not None:
            if not self.send_resource_prediction(results["resource_prediction"], location):
                success = False
        
        return success
# API URL을 환경 변수에서 가져오는 함수 
def get_api_url():
    """
    환경 변수에서 API URL 가져오기
    
    Returns:
        str: API URL
    """
    import os
    return os.environ.get("API_URL", "http://localhost:10272/ai/data")