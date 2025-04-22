#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터 전처리 유틸리티 모듈

원시 데이터를 모델 학습에 적합한 형태로 변환하는 함수들 제공
데이터 정규화, 특성 생성, 이상치 처리 등의 기능 수행
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

# 로깅 설정
logger = logging.getLogger(__name__)

def create_cpu_features(df):
    """
    CPU 데이터에서 특성 생성
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    result_df = df.copy()
    
    # CPU 관련 컬럼 검색
    cpu_cols = [col for col in df.columns if any(kw in col.lower() for kw in 
                                               ['usage_user', 'usage_system', 'usage_idle', 'usage_iowait'])]
    
    if not cpu_cols:
        logger.info("CPU 관련 컬럼을 찾을 수 없습니다.")
        return result_df
    
    try:
        # 1. CPU 총 사용률 (user + system)
        if 'usage_user' in df.columns and 'usage_system' in df.columns:
            result_df['cpu_total_usage'] = df['usage_user'] + df['usage_system']
        
        # 2. CPU 부하율 (100 - idle)
        if 'usage_idle' in df.columns:
            result_df['cpu_load'] = 100 - df['usage_idle']
        
        # 3. I/O 대기 비율 변화
        if 'usage_iowait' in df.columns:
            result_df['iowait_change'] = df['usage_iowait'].diff().fillna(0)
            
        # 4. CPU 사용률 기울기 (추세)
        for col in cpu_cols:
            # 5분 이동 평균
            result_df[f'{col}_ma5'] = df[col].rolling(window=5, min_periods=1).mean()
            
            # 변화율
            result_df[f'{col}_change'] = df[col].pct_change().fillna(0)
            
            # 급증 여부 (전 시간 대비 20% 이상 증가)
            result_df[f'{col}_surge'] = (df[col].pct_change() > 0.2).astype(int)

        logger.info(f"CPU 특성 {len(result_df.columns) - len(df.columns)}개 생성됨")
        return result_df
    
    except Exception as e:
        logger.error(f"CPU 특성 생성 중 오류 발생: {e}")
        return df

def create_memory_features(df):
    """
    메모리 데이터에서 특성 생성
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    result_df = df.copy()
    
    # 메모리 관련 컬럼 검색
    mem_cols = [col for col in df.columns if any(kw in col.lower() for kw in 
                                               ['mem', 'memory', 'used_percent', 'available'])]
    
    if not mem_cols:
        logger.info("메모리 관련 컬럼을 찾을 수 없습니다.")
        return result_df
    
    try:
        # 1. 메모리 사용률 변화
        for col in mem_cols:
            if 'used' in col.lower() or 'available' in col.lower():
                # 변화율
                result_df[f'{col}_change'] = df[col].diff().fillna(0)
                
                # 5분 이동 평균
                result_df[f'{col}_ma5'] = df[col].rolling(window=5, min_periods=1).mean()
                
                # 급증 여부
                result_df[f'{col}_surge'] = (df[col].diff() > df[col].std()).astype(int)
        
        # 2. 메모리 임계치 접근도 (90% 기준)
        for col in mem_cols:
            if 'used' in col.lower():
                result_df[f'{col}_threshold_90'] = (df[col] / 90).clip(upper=1.0)
        
        logger.info(f"메모리 특성 {len(result_df.columns) - len(df.columns)}개 생성됨")
        return result_df
    
    except Exception as e:
        logger.error(f"메모리 특성 생성 중 오류 발생: {e}")
        return df

def create_disk_features(df):
    """
    디스크 데이터에서 특성 생성
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    result_df = df.copy()
    
    # 디스크 관련 컬럼 검색
    disk_cols = [col for col in df.columns if any(kw in col.lower() for kw in 
                                                ['disk', 'io', 'read', 'write'])]
    
    if not disk_cols:
        logger.info("디스크 관련 컬럼을 찾을 수 없습니다.")
        return result_df
    
    try:
        # 1. 디스크 사용률 변화
        for col in disk_cols:
            if 'used' in col.lower():
                # 변화율
                result_df[f'{col}_change'] = df[col].diff().fillna(0)
                
                # 5분 이동 평균
                result_df[f'{col}_ma5'] = df[col].rolling(window=5, min_periods=1).mean()
                
                # 임계치 접근도 (90% 기준)
                result_df[f'{col}_threshold_90'] = (df[col] / 90).clip(upper=1.0)
        
        # 2. I/O 활동 지표
        io_cols = [col for col in disk_cols if any(kw in col.lower() for kw in ['io', 'read', 'write'])]
        for col in io_cols:
            # 변화율
            result_df[f'{col}_change'] = df[col].pct_change().fillna(0)
            
            # 이동 평균
            result_df[f'{col}_ma5'] = df[col].rolling(window=5, min_periods=1).mean()
            
            # 급증 여부
            result_df[f'{col}_surge'] = (df[col].pct_change() > 0.5).astype(int)
        
        logger.info(f"디스크 특성 {len(result_df.columns) - len(df.columns)}개 생성됨")
        return result_df
    
    except Exception as e:
        logger.error(f"디스크 특성 생성 중 오류 발생: {e}")
        return df

def create_network_features(df):
    """
    네트워크 데이터에서 특성 생성
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    result_df = df.copy()
    
    # 네트워크 관련 컬럼 검색
    net_cols = [col for col in df.columns if any(kw in col.lower() for kw in 
                                               ['net', 'bytes_sent', 'bytes_recv', 'drop', 'err'])]
    
    if not net_cols:
        logger.info("네트워크 관련 컬럼을 찾을 수 없습니다.")
        return result_df
    
    try:
        # 1. 트래픽 변화
        traffic_cols = [col for col in net_cols if any(kw in col.lower() for kw in ['bytes_sent', 'bytes_recv'])]
        for col in traffic_cols:
            # 변화율
            result_df[f'{col}_change'] = df[col].pct_change().fillna(0)
            
            # 이동 평균
            result_df[f'{col}_ma5'] = df[col].rolling(window=5, min_periods=1).mean()
            
            # 급증 여부
            result_df[f'{col}_surge'] = (df[col].pct_change() > 0.5).astype(int)
        
        # 2. 오류/드롭 변화
        error_cols = [col for col in net_cols if any(kw in col.lower() for kw in ['drop', 'err'])]
        for col in error_cols:
            # 증가 여부
            result_df[f'{col}_increase'] = (df[col].diff() > 0).astype(int)
            
            # 누적 차이
            result_df[f'{col}_cum_diff'] = df[col].diff().cumsum().fillna(0)
        
        # 3. 송수신 비율 (가능한 경우)
        if any('bytes_sent' in col.lower() for col in net_cols) and any('bytes_recv' in col.lower() for col in net_cols):
            sent_col = next(col for col in net_cols if 'bytes_sent' in col.lower())
            recv_col = next(col for col in net_cols if 'bytes_recv' in col.lower())
            
            # 송수신 비율
            result_df['net_send_recv_ratio'] = (df[sent_col] / df[recv_col].clip(lower=1)).fillna(1)
        
        logger.info(f"네트워크 특성 {len(result_df.columns) - len(df.columns)}개 생성됨")
        return result_df
    
    except Exception as e:
        logger.error(f"네트워크 특성 생성 중 오류 발생: {e}")
        return df

def create_system_features(df):
    """
    시스템 데이터에서 특성 생성
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    result_df = df.copy()
    
    # 시스템 관련 컬럼 검색
    sys_cols = [col for col in df.columns if any(kw in col.lower() for kw in 
                                               ['load', 'system', 'uptime'])]
    
    if not sys_cols:
        logger.info("시스템 관련 컬럼을 찾을 수 없습니다.")
        return result_df
    
    try:
        # 1. 시스템 로드 변화
        load_cols = [col for col in sys_cols if 'load' in col.lower()]
        for col in load_cols:
            # 변화율
            result_df[f'{col}_change'] = df[col].diff().fillna(0)
            
            # 이동 평균
            result_df[f'{col}_ma5'] = df[col].rolling(window=5, min_periods=1).mean()
            
            # 로드 급증 여부 (기준: 전 시간 대비 0.5 이상 증가)
            result_df[f'{col}_surge'] = (df[col].diff() > 0.5).astype(int)
            
            # 코어 수 기준 로드 비율 (일반적인 코어 수 가정)
            cpu_cores = 4  # 기본값으로 4코어 가정
            result_df[f'{col}_per_core'] = df[col] / cpu_cores
            
            # 높은 로드 상태
            result_df[f'{col}_high'] = (df[col] > cpu_cores * 0.8).astype(int)
        
        logger.info(f"시스템 특성 {len(result_df.columns) - len(df.columns)}개 생성됨")
        return result_df
    
    except Exception as e:
        logger.error(f"시스템 특성 생성 중 오류 발생: {e}")
        return df

def create_temp_features(df):
    """
    온도 데이터에서 특성 생성
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    result_df = df.copy()
    
    # 온도 관련 컬럼 검색
    temp_cols = [col for col in df.columns if any(kw in col.lower() for kw in 
                                                ['temp', 'temperature'])]
    
    if not temp_cols:
        logger.info("온도 관련 컬럼을 찾을 수 없습니다.")
        return result_df
    
    try:
        for col in temp_cols:
            # 변화율
            result_df[f'{col}_change'] = df[col].diff().fillna(0)
            
            # 이동 평균
            result_df[f'{col}_ma5'] = df[col].rolling(window=5, min_periods=1).mean()
            
            # 이동 표준편차 (변동성)
            result_df[f'{col}_std5'] = df[col].rolling(window=5, min_periods=1).std().fillna(0)
            
            # 임곗값 초과 여부 (일반적으로 70도 이상은 위험)
            result_df[f'{col}_high'] = (df[col] > 70).astype(int)
            
            # 급증 여부 (이전 평균보다 5도 이상 증가)
            baseline = df[col].shift(1).rolling(window=5, min_periods=1).mean()
            result_df[f'{col}_spike'] = (df[col] > baseline + 5).astype(int)
        
        logger.info(f"온도 특성 {len(result_df.columns) - len(df.columns)}개 생성됨")
        return result_df
    
    except Exception as e:
        logger.error(f"온도 특성 생성 중 오류 발생: {e}")
        return df

def create_time_features(df):
    """
    시간 기반 특성 생성
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임 (인덱스가 시간)
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    result_df = df.copy()
    
    # 인덱스가 시간 기반인지 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.info("데이터프레임 인덱스가 DatetimeIndex가 아닙니다.")
        return result_df
    
    try:
        # 시간 기반 특성 생성
        # 1. 시간 관련
        result_df['hour'] = df.index.hour
        result_df['day_of_week'] = df.index.dayofweek
        result_df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # 2. 업무 시간
        result_df['is_business_hour'] = ((df.index.hour >= 9) & 
                                       (df.index.hour < 18) & 
                                       (df.index.dayofweek < 5)).astype(int)
        
        # 3. 피크 시간 (일반적으로 오전 9시-11시, 오후 1시-5시)
        result_df['is_peak_hour'] = ((df.index.hour >= 9) & (df.index.hour <= 11) | 
                                   (df.index.hour >= 13) & (df.index.hour <= 17)).astype(int)
        
        # 4. 계절 정보 (북반구 기준)
        month = df.index.month
        result_df['is_winter'] = ((month == 12) | (month == 1) | (month == 2)).astype(int)
        result_df['is_spring'] = ((month >= 3) & (month <= 5)).astype(int)
        result_df['is_summer'] = ((month >= 6) & (month <= 8)).astype(int)
        result_df['is_fall'] = ((month >= 9) & (month <= 11)).astype(int)
        
        # 5. 날짜 주기성을 사인/코사인으로 표현
        # 시간 주기 (24시간)
        hours = df.index.hour
        result_df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        
        # 요일 주기 (7일)
        weekday = df.index.dayofweek
        result_df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
        result_df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
        
        logger.info(f"시간 특성 {len(result_df.columns) - len(df.columns)}개 생성됨")
        return result_df
    
    except Exception as e:
        logger.error(f"시간 특성 생성 중 오류 발생: {e}")
        return df

def create_correlation_features(df):
    """
    여러 측정 항목 간의 상관관계 특성 생성
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        
    Returns:
        pd.DataFrame: 특성이 추가된 데이터프레임
    """
    result_df = df.copy()
    
    try:
        # 1. CPU-메모리 관계
        if 'cpu_load' in df.columns and 'mem_usedPercent_used_percent' in df.columns:
            # CPU/메모리 비율
            result_df['cpu_mem_ratio'] = (df['cpu_load'] / 
                                        df['mem_usedPercent_used_percent'].clip(lower=0.1))
            
            # CPU와 메모리가 모두 높은 경우
            result_df['high_cpu_mem'] = ((df['cpu_load'] > 80) & 
                                       (df['mem_usedPercent_used_percent'] > 80)).astype(int)
        
        # 2. CPU-I/O 관계
        if 'usage_iowait' in df.columns and 'usage_user' in df.columns:
            # I/O 대기 비율 (I/O 대기 / 사용자 사용률)
            result_df['io_cpu_ratio'] = (df['usage_iowait'] / 
                                       df['usage_user'].clip(lower=0.1))
            
            # I/O 병목 지표 (사용률은 낮은데 I/O 대기는 높은 경우)
            result_df['io_bottleneck'] = ((df['usage_user'] < 50) & 
                                        (df['usage_iowait'] > 20)).astype(int)
        
        # 3. CPU-네트워크 관계
        net_cols = [col for col in df.columns if any(kw in col.lower() for kw in 
                                                  ['bytes_sent', 'bytes_recv'])]
        if 'cpu_load' in df.columns and net_cols:
            # 네트워크 활동 중 CPU 사용률
            for col in net_cols:
                high_net = df[col] > df[col].quantile(0.75)  # 상위 25% 네트워크 활동
                result_df[f'high_{col}_cpu'] = (high_net & (df['cpu_load'] > 70)).astype(int)
        
        # 4. 메모리-디스크 관계
        if 'mem_usedPercent_used_percent' in df.columns and 'disk_usedPercent_used_percent' in df.columns:
            # 모두 높은 경우 (저장 공간 부족 위험)
            result_df['high_mem_disk'] = ((df['mem_usedPercent_used_percent'] > 80) & 
                                        (df['disk_usedPercent_used_percent'] > 80)).astype(int)
        
        # 5. 온도-CPU 관계
        if 'temp_input' in df.columns and 'cpu_load' in df.columns:
            # CPU 사용률과 온도 상관관계
            result_df['temp_cpu_ratio'] = df['temp_input'] / (df['cpu_load'].clip(lower=1))
            
            # 고부하-고온 상태
            result_df['high_temp_cpu'] = ((df['temp_input'] > 70) & 
                                        (df['cpu_load'] > 80)).astype(int)
        
        logger.info(f"상관관계 특성 {len(result_df.columns) - len(df.columns)}개 생성됨")
        return result_df
    
    except Exception as e:
        logger.error(f"상관관계 특성 생성 중 오류 발생: {e}")
        return df

def remove_outliers(df, cols=None, method='z-score', threshold=3.0):
    """
    이상치 제거 또는 대체
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        cols (list): 처리할 열 목록 (None이면 모든 숫자형 열)
        method (str): 이상치 감지 방법 ('z-score', 'iqr')
        threshold (float): 이상치 감지 임계값
        
    Returns:
        pd.DataFrame: 이상치가 처리된 데이터프레임
    """
    result_df = df.copy()
    
    # 처리할 열 선택
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns
    else:
        cols = [col for col in cols if col in df.columns]
    
    if not cols:
        logger.info("처리할 숫자형 열이 없습니다.")
        return result_df
    
    try:
        for col in cols:
            # Z-점수 방식
            if method == 'z-score':
                z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                outliers = z_scores > threshold
                
            # IQR 방식
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            else:
                logger.warning(f"지원하지 않는 이상치 감지 방법: {method}")
                continue
            
            # 이상치 대체 (중앙값으로)
            if outliers.sum() > 0:
                median_val = df[col].median()
                result_df.loc[outliers, col] = median_val
                logger.info(f"'{col}' 열에서 {outliers.sum()}개 이상치를 {median_val}로 대체")
        
        return result_df
        
    except Exception as e:
        logger.error(f"이상치 처리 중 오류 발생: {e}")
        return df

def normalize_features(df, cols=None, method='minmax'):
    """
    특성 정규화
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        cols (list): 처리할 열 목록 (None이면 모든 숫자형 열)
        method (str): 정규화 방법 ('minmax', 'standard')
        
    Returns:
        tuple: (정규화된 데이터프레임, 스케일러 객체 딕셔너리)
    """
    result_df = df.copy()
    
    # 처리할 열 선택
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns
    else:
        cols = [col for col in cols if col in df.columns]
    
    if not cols:
        logger.info("정규화할 숫자형 열이 없습니다.")
        return result_df, {}
    
    try:
        scalers = {}
        
        for col in cols:
            # MinMax 정규화 (0~1 범위)
            if method == 'minmax':
                scaler = MinMaxScaler()
                
            # Z-점수 정규화 (평균 0, 표준편차 1)
            elif method == 'standard':
                scaler = StandardScaler()
                
            else:
                logger.warning(f"지원하지 않는 정규화 방법: {method}")
                continue
            
            # 2D 배열 형태로 변환하여 스케일링
            result_df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
            scalers[col] = scaler
        
        logger.info(f"{len(cols)}개 열에 {method} 정규화 적용 완료")
        return result_df, scalers
        
    except Exception as e:
        logger.error(f"특성 정규화 중 오류 발생: {e}")
        return df, {}

def preprocess_data_advanced(df, target_col=None):
    """
    데이터 고급 전처리
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        target_col (str): 타겟 열 이름
        
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    if df.empty:
        logger.warning("전처리할 데이터가 비어 있습니다.")
        return df
    
    try:
        # 1. 결측치 처리
        if df.isna().sum().sum() > 0:
            df = df.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
            logger.info("결측치 보간 완료")
        
        # 2. 이상치 처리
        df = remove_outliers(df, method='z-score', threshold=3.0)
        
        # 3. 시간 기반 특성 생성
        if isinstance(df.index, pd.DatetimeIndex):
            df = create_time_features(df)
        
        # 4. 리소스 기반 특성 생성
        df = create_cpu_features(df)
        df = create_memory_features(df)
        df = create_disk_features(df)
        df = create_network_features(df)
        df = create_system_features(df)
        df = create_temp_features(df)
        
        # 5. 상관관계 기반 특성 생성
        df = create_correlation_features(df)
        
        # 결측치 제거 (최종)
        df = df.fillna(0)
        
        logger.info(f"전처리 완료: {len(df)}행, {df.shape[1]}열")
        return df
        
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {e}")
        return df

def prepare_data_for_model(df, target_col, test_size=0.2, normalize=True):
    """
    모델 학습을 위한 데이터 준비
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        target_col (str): 타겟 열 이름
        test_size (float): 테스트 세트 비율
        normalize (bool): 정규화 여부
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scalers)
    """
    if df.empty:
        logger.warning("준비할 데이터가 비어 있습니다.")
        return None, None, None, None, None
    
    if target_col not in df.columns:
        logger.error(f"타겟 열 '{target_col}'이 데이터에 없습니다.")
        return None, None, None, None, None
    
    try:
        # 특성과 타겟 분리
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 오래된 데이터를 기준으로 시간순 분할
        train_size = int(len(df) * (1 - test_size))
        
        if isinstance(df.index, pd.DatetimeIndex):
            # 시간 인덱스가 있는 경우 시간순 분할
            X_train = X.iloc[:train_size]
            X_test = X.iloc[train_size:]
            y_train = y.iloc[:train_size]
            y_test = y.iloc[train_size:]
        else:
            # 일반적인 랜덤 분할
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # 정규화
        scalers = {}
        if normalize:
            X_train_norm, scalers = normalize_features(X_train, method='minmax')
            
            # 테스트 데이터에 같은 스케일러 적용
            X_test_norm = X_test.copy()
            for col in X_test.columns:
                if col in scalers:
                    X_test_norm[col] = scalers[col].transform(
                        X_test[col].values.reshape(-1, 1)
                    ).flatten()
            
            return X_train_norm, X_test_norm, y_train, y_test, scalers
        
        return X_train, X_test, y_train, y_test, scalers
        
    except Exception as e:
        logger.error(f"모델 데이터 준비 중 오류 발생: {e}")
        return None, None, None, None, None

def detect_anomalies(df, method='isolation_forest', contamination=0.05):
    """
    이상 탐지
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        method (str): 이상 탐지 방법 ('isolation_forest', 'one_class_svm', 'dbscan')
        contamination (float): 예상 이상치 비율
        
    Returns:
        pd.Series: 이상치 여부 (True/False)
    """
    if df.empty:
        logger.warning("이상 탐지할 데이터가 비어 있습니다.")
        return pd.Series([], dtype=bool)
    
    try:
        # 숫자형 데이터만 사용
        X = df.select_dtypes(include=np.number)
        
        if X.empty:
            logger.warning("이상 탐지할 숫자형 데이터가 없습니다.")
            return pd.Series(False, index=df.index)
        
        # 결측치 처리
        X = X.fillna(X.mean())
        
        # Isolation Forest
        if method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
        
        # One-Class SVM
        elif method == 'one_class_svm':
            from sklearn.svm import OneClassSVM
            model = OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='scale'
            )
        
        # DBSCAN
        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            
            X_scaled = StandardScaler().fit_transform(X)
            
            # DBSCAN은 별도의 방식으로 호출
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X_scaled)
            
            # -1은 이상치를 의미
            anomalies = pd.Series(labels == -1, index=df.index)
            logger.info(f"DBSCAN 이상 탐지 완료: {anomalies.sum()}개 이상치 발견")
            return anomalies
        
        else:
            logger.warning(f"지원하지 않는 이상 탐지 방법: {method}")
            return pd.Series(False, index=df.index)
        
        # 모델 학습 및 예측
        predictions = model.fit_predict(X)
        
        # Isolation Forest와 One-Class SVM은 -1이 이상치, 1이 정상치
        anomalies = pd.Series(predictions == -1, index=df.index)
        
        logger.info(f"{method} 이상 탐지 완료: {anomalies.sum()}개 이상치 발견")
        return anomalies
        
    except Exception as e:
        logger.error(f"이상 탐지 중 오류 발생: {e}")
        return pd.Series(False, index=df.index)

def extract_features_importance(df, target_col, method='random_forest'):
    """
    특성 중요도 계산
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        target_col (str): 타겟 열 이름
        method (str): 중요도 계산 방법 ('random_forest', 'xgboost', 'permutation')
        
    Returns:
        pd.DataFrame: 특성별 중요도
    """
    if df.empty:
        logger.warning("특성 중요도를 계산할 데이터가 비어 있습니다.")
        return pd.DataFrame()
    
    if target_col not in df.columns:
        logger.error(f"타겟 열 '{target_col}'이 데이터에 없습니다.")
        return pd.DataFrame()
    
    try:
        # 특성과 타겟 분리
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 범주형 데이터 숫자형으로 변환 (간단히)
        X = X.select_dtypes(include=np.number)
        
        # 결측치 처리
        X = X.fillna(X.mean())
        
        # 랜덤 포레스트
        if method == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            importances = model.feature_importances_
            
        # XGBoost
        elif method == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            importances = model.feature_importances_
            
        # 순열 중요도
        elif method == 'permutation':
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance
            
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            result = permutation_importance(
                model, X, y, n_repeats=10, random_state=42
            )
            importances = result.importances_mean
            
        else:
            logger.warning(f"지원하지 않는 특성 중요도 계산 방법: {method}")
            return pd.DataFrame()
        
        # 중요도 데이터프레임 생성
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        })
        
        # 중요도 기준 내림차순 정렬
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info(f"{method} 특성 중요도 계산 완료")
        return importance_df
        
    except Exception as e:
        logger.error(f"특성 중요도 계산 중 오류 발생: {e}")
        return pd.DataFrame()

def preprocess_for_lstm(df, target_col=None, sequence_length=24, normalization='global'):
    """
    LSTM 모델을 위한 시계열 데이터 전처리
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        target_col (str): 타겟 열 이름 (None이면 모든 열 사용)
        sequence_length (int): 시퀀스 길이
        normalization (str): 정규화 방법 ('global', 'local', 'none')
        
    Returns:
        tuple: (X, y, scalers)
    """
    if df.empty:
        logger.warning("전처리할 데이터가 비어 있습니다.")
        return None, None, None
    
    try:
        # 타겟 열이 없으면 모든 열 사용
        if target_col is None:
            data = df.select_dtypes(include=np.number)
        else:
            if target_col not in df.columns:
                logger.error(f"타겟 열 '{target_col}'이 데이터에 없습니다.")
                return None, None, None
            data = df[[target_col]]
        
        # 결측치 처리
        data = data.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
        
        # 정규화
        scalers = {}
        
        if normalization == 'global':
            # 전체 데이터에 대해 정규화
            scaled_data, scalers = normalize_features(data, method='minmax')
            
        elif normalization == 'local':
            # 각 시퀀스별로 정규화 (스케일러는 반환하지 않음)
            scaled_data = data.copy()
            for i in range(len(data) - sequence_length):
                seq = data.iloc[i:i+sequence_length]
                seq_norm, _ = normalize_features(seq, method='minmax')
                scaled_data.iloc[i:i+sequence_length] = seq_norm
            
        else:  # 'none'
            scaled_data = data
        
        # 시퀀스 생성
        X, y = [], []
        
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data.iloc[i:i+sequence_length].values)
            if target_col is not None:
                y.append(scaled_data.iloc[i+sequence_length][target_col])
            else:
                y.append(scaled_data.iloc[i+sequence_length].values)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"LSTM 전처리 완료: X 형태 {X.shape}, y 형태 {y.shape}")
        return X, y, scalers
        
    except Exception as e:
        logger.error(f"LSTM 전처리 중 오류 발생: {e}")
        return None, None, None