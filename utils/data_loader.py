#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터 로드 및 전처리 유틸리티

InfluxDB, MySQL, CSV 등 다양한 데이터 소스에서 데이터를 로드하고 전처리하는 기능 제공
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

# 로깅 설정
logger = logging.getLogger(__name__)

# 데이터 파일 경로 정의 (data 폴더 내)
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

def load_csv_data(file_path):
    """
    CSV 파일에서 데이터 로드
    
    Args:
        file_path (str): CSV 파일 경로
        
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    logger.info(f"CSV 파일 로드 중: {file_path}")
    
    try:
        # 파일 존재 확인
        if not os.path.exists(file_path):
            logger.error(f"CSV 파일이 존재하지 않습니다: {file_path}")
            return pd.DataFrame()
        
        # 파일 크기 확인
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            logger.error(f"CSV 파일이 비어 있습니다: {file_path}")
            return pd.DataFrame()
        
        # 파일 로드
        df = pd.read_csv(file_path)
        
        # 첫 번째 열이 날짜/시간인지 확인
        first_col = df.columns[0]
        
        try:
            # 타임스탬프 열 감지 및 변환
            time_col = None
            
            # 일반적인 시간 관련 열 이름
            time_cols = ['timestamp', 'time', 'date', 'datetime', first_col]
            
            for col in time_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        time_col = col
                        break
                    except:
                        continue
            
            # 타임스탬프 열을 인덱스로 설정
            if time_col:
                df = df.set_index(time_col)
                logger.info(f"타임스탬프 열 '{time_col}'을 인덱스로 설정했습니다.")
            else:
                logger.warning("타임스탬프 열을 찾을 수 없습니다.")
                # 인덱스가 숫자인 경우 임의로 타임스탬프 생성
                if isinstance(df.index, pd.RangeIndex):
                    start_time = datetime.now() - timedelta(hours=len(df)-1)
                    time_index = pd.date_range(start=start_time, periods=len(df), freq='H')
                    df.index = time_index
                    logger.info("임의의 시간 인덱스를 생성했습니다.")
            
            # 모든 숫자형 열 추출
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # 숫자형 열만 남기기
            if len(numeric_cols) > 0:
                df = df[numeric_cols]
                logger.info(f"숫자형 열 {len(numeric_cols)}개 추출: {numeric_cols.tolist()}")
            else:
                logger.warning("숫자형 열이 없습니다.")
            
            # 결측값 처리
            if df.isna().sum().sum() > 0:
                df = df.interpolate(method='time')
                logger.info("결측값을 보간했습니다.")
            
            logger.info(f"CSV 로드 완료: {len(df)}개 행, {df.shape[1]}개 열")
            return df
            
        except Exception as e:
            logger.error(f"CSV 전처리 중 오류 발생: {e}")
            return df  # 원본 데이터프레임 반환
        
    except Exception as e:
        logger.error(f"CSV 파일 로드 중 오류 발생: {e}")
        return pd.DataFrame()

def load_all_data():
    """
    모든 데이터 파일을 로드하고 통합
    
    Returns:
        pd.DataFrame: 통합된 데이터프레임
    """
    all_dfs = []
    
    for category, files in DATA_FILES.items():
        logger.info(f"{category} 데이터 로드 중...")
        category_dfs = []
        
        for file_path in files:
            if os.path.exists(file_path):
                df = load_influxdb_csv_data(file_path)
                if not df.empty:
                    logger.info(f"  - {file_path}: {len(df)}행, {df.shape[1]}열")
                    
                    # 각 데이터프레임의 열 이름을 파일 이름에 기반하여 고유하게 만듦
                    file_name = os.path.basename(file_path).replace('.csv', '')
                    df.columns = [f"{file_name}_{col}" for col in df.columns]
                    
                    category_dfs.append(df)
                else:
                    logger.warning(f"  - {file_path}: 빈 데이터프레임")
            else:
                logger.warning(f"  - {file_path}: 파일이 존재하지 않음")
        
        if category_dfs:
            # 모든 데이터프레임을 한 번에 결합
            # join='outer'를 사용하여 모든 타임스탬프 포함
            try:
                if len(category_dfs) == 1:
                    category_df = category_dfs[0]
                else:
                    # 인덱스 기준으로 조인 (outer join)
                    category_df = category_dfs[0]
                    for df in category_dfs[1:]:
                        category_df = category_df.join(df, how='outer')
                
                all_dfs.append(category_df)
            except Exception as e:
                logger.error(f"{category} 데이터 병합 중 오류: {e}")
    
    if not all_dfs:
        logger.error("로드된 데이터가 없습니다.")
        return pd.DataFrame()
    
    # 모든 카테고리 데이터프레임 병합
    try:
        if len(all_dfs) == 1:
            final_df = all_dfs[0]
        else:
            # 여러 카테고리를 한 번에 조인
            final_df = all_dfs[0]
            for df in all_dfs[1:]:
                final_df = final_df.join(df, how='outer')
        
        # 중복 컬럼 제거
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        
        # 시간 순으로 정렬
        final_df = final_df.sort_index()
        
        # 데이터 정리
        final_df = final_df.interpolate(method='time')
        final_df = final_df.dropna(how='all')
        
        # 컬럼 이름 간소화 (필요한 경우)
        new_columns = []
        for col in final_df.columns:
            # 파일 이름 접두사 제거
            if '_usage_' in col:
                new_col = col.split('_usage_')[1]
            elif '_temp_' in col:
                new_col = 'temp_input'
            elif '_load' in col:
                new_col = col.split('_')[-1]
            else:
                new_col = col
            
            # 중복 방지
            if new_col in new_columns:
                new_col = col  # 원래 이름 유지
            new_columns.append(new_col)
        
        final_df.columns = new_columns
        
        logger.info(f"최종 통합 데이터프레임: {len(final_df)}행, {final_df.shape[1]}열")
        logger.info(f"컬럼: {final_df.columns.tolist()}")
        
        return final_df
        
    except Exception as e:
        logger.error(f"최종 데이터 병합 중 오류: {e}")
        return pd.DataFrame()

def query_influxdb(config, start_time, origins=None):
    """
    InfluxDB에서 데이터 쿼리 (origin 태그 기반)
    
    Args:
        config (dict): InfluxDB 설정
        start_time (datetime): 쿼리 시작 시간
        origins (list): 쿼리할 origin 목록 (None이면 모든 origin)
        
    Returns:
        pd.DataFrame: 쿼리 결과
    """
    influx_config = config["influxdb"]
    
    # origin 목록 처리
    if origins is None:
        origins = ["server_data", "sensor_data"]  # 기본값
    elif isinstance(origins, str):
        origins = [origins]
    
    try:
        # 로그에 설정 정보 출력
        masked_token = influx_config["token"][:4] + "..." if len(influx_config["token"]) > 4 else "***"
        logger.info(f"InfluxDB 연결 시도: URL={influx_config['url']}, Org={influx_config['org']}, Bucket={influx_config['bucket']}, Token={masked_token}")
        logger.info(f"쿼리 대상 origins: {origins}")
        
        client = InfluxDBClient(
            url=influx_config["url"],
            token=influx_config["token"],
            org=influx_config["org"]
        )
        
        query_api = client.query_api()
        
        # 시작 시간 문자열 형식으로 변환
        start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # 모든 origin에 대한 쿼리 결과를 담을 DataFrame
        all_data = pd.DataFrame()
        
        # 각 origin마다 개별 쿼리 실행
        for origin in origins:
            logger.info(f"'{origin}' origin 쿼리 실행")
            
            # pivot 쿼리 사용하여 필드를 컬럼으로 변환
            query = f'''
            from(bucket: "{influx_config["bucket"]}")
              |> range(start: {start_time_str})
              |> filter(fn: (r) => r["origin"] == "{origin}")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            try:
                # 데이터프레임으로 직접 쿼리
                result = query_api.query_data_frame(query)
                
                if isinstance(result, list):
                    # 여러 테이블이 반환된 경우 병합
                    result = pd.concat(result)
                
                if result.empty:
                    logger.warning(f"'{origin}' origin에서 데이터가 없습니다.")
                    continue
                
                # _time을 인덱스로 설정
                if '_time' in result.columns:
                    result.set_index('_time', inplace=True)
                
                # 불필요한 메타데이터 컬럼 제거
                exclude_cols = ['result', 'table', '_start', '_stop', '_measurement']
                data_cols = [col for col in result.columns if col not in exclude_cols]
                
                # origin 식별을 위한 접두어 추가
                result = result[data_cols]
                if origin == "server_data":
                    # 서버 데이터는 원래 이름 유지
                    origin_df = result
                else:
                    # 센서 데이터는 접두어 추가
                    origin_df = result.add_prefix(f"{origin}_")
                
                # 전체 결과에 추가
                if all_data.empty:
                    all_data = origin_df
                else:
                    all_data = all_data.join(origin_df, how='outer')
                
                logger.info(f"'{origin}' origin 쿼리 결과: {len(origin_df)}개 행, {origin_df.shape[1]}개 열")
            
            except Exception as e:
                logger.error(f"'{origin}' origin 쿼리 중 오류: {e}")
                continue
                
        # 결과가 비어있는지 확인
        if all_data.empty:
            logger.warning(f"모든 origin에 대한 쿼리 결과가 비어있습니다.")
            return pd.DataFrame()
        
        # 결측치 처리
        all_data = all_data.interpolate(method='time')
        
        logger.info(f"InfluxDB 최종 쿼리 결과: {len(all_data)}개 행, {all_data.shape[1]}개 열")
        logger.info(f"컬럼: {list(all_data.columns)[:10]}{'...' if len(all_data.columns) > 10 else ''}")
        
        return all_data
        
    except Exception as e:
        logger.error(f"InfluxDB 쿼리 중 오류 발생: {e}")
        return pd.DataFrame()
    
    finally:
        if 'client' in locals():
            client.close()

def load_influxdb_csv_data(csv_path):
    """
    InfluxDB 형식의 CSV 파일을 직접 로드하고 처리
    
    Args:
        csv_path (str): CSV 파일 경로
        
    Returns:
        pd.DataFrame: 처리된 데이터프레임
    """
    logger.info(f"InfluxDB CSV 데이터 파일 로드 중: {csv_path}")
    
    try:
        # 파일 존재 확인
        if not os.path.exists(csv_path):
            logger.error(f"CSV 파일이 존재하지 않습니다: {csv_path}")
            return pd.DataFrame()
        
        # 파일 크기 확인
        file_size = os.path.getsize(csv_path)
        if file_size == 0:
            logger.error(f"CSV 파일이 비어 있습니다: {csv_path}")
            return pd.DataFrame()
        
        # InfluxDB CSV는 처음 3줄이 메타데이터임
        df = pd.read_csv(csv_path, skiprows=3)
        
        # 컬럼 이름 정리
        df.columns = [col.strip() for col in df.columns]
        
        # 필요한 컬럼 확인
        required_columns = ['_time', '_value', '_field', '_measurement']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"필수 컬럼이 누락되었습니다: {missing_columns}")
            return pd.DataFrame()
        
        # 타임스탬프 변환
        try:
            df['_time'] = pd.to_datetime(df['_time'])
        except Exception as e:
            logger.warning(f"시간 변환 중 오류: {e}, 대체 방법 시도")
            
            # 대체 방법: 나노초 부분을 제거하고 파싱
            df['_time'] = pd.to_datetime(df['_time'].astype(str).str.replace(r'\.\d+Z$', 'Z', regex=True))
        
        # 숫자형 값으로 변환
        df['_value'] = pd.to_numeric(df['_value'], errors='coerce')
        
        # 결측치 제거
        df = df.dropna(subset=['_value'])
        
        # 인덱스 설정
        df.set_index('_time', inplace=True)
        
        # 중복 인덱스 제거
        df = df[~df.index.duplicated(keep='first')]
        
        # 시간 순으로 정렬
        df = df.sort_index()
        
        # 필드별로 피벗 (여러 필드가 있는 경우)
        if '_field' in df.columns and df['_field'].nunique() > 1:
            # 각 필드를 별도 컬럼으로 변환
            df_pivot = df.pivot_table(
                index=df.index,
                columns='_field',
                values='_value',
                aggfunc='first'
            )
            return df_pivot
        else:
            # 단일 필드인 경우 그대로 반환
            if '_field' in df.columns and df['_field'].nunique() == 1:
                field_name = df['_field'].iloc[0]
                return df[['_value']].rename(columns={'_value': field_name})
            else:
                return df[['_value']]
            
    except Exception as e:
        logger.error(f"CSV 파일 로드 중 예상치 못한 오류: {e}")
        return pd.DataFrame()

def generate_test_data(days=30, hour_interval=1, start_time=None, cols=None):
    """
    테스트용 가상 IoT 센서 데이터 생성
    
    Args:
        days (int): 생성할 데이터 기간(일)
        hour_interval (int): 데이터 포인트 간격(시간)
        start_time (datetime): 시작 시간 (None이면 현재시간-days)
        cols (list): 생성할 열 이름 목록 (None이면 기본값 사용)
        
    Returns:
        pd.DataFrame: 생성된 테스트 데이터
    """
    logger.info(f"테스트 데이터 생성 중 ({days}일, {hour_interval}시간 간격)...")
    
    # 시간 범위 생성
    if start_time is None:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
    else:
        end_time = start_time + timedelta(days=days)
        
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{hour_interval}H')
    
    # 기본 열 이름 및 기본값 (평균, 표준편차)
    default_cols = {
        'usage_user': (40, 15),       # CPU 사용자 사용률 (%)
        'usage_system': (10, 5),      # CPU 시스템 사용률 (%)
        'usage_iowait': (5, 3),       # CPU I/O 대기 비율 (%)
        'usage_idle': (45, 20),       # CPU 유휴 비율 (%)
        'temp_input': (45, 10),       # 온도 (°C)
        'mem_used_percent': (60, 15), # 메모리 사용률 (%)
        'disk_used_percent': (70, 5), # 디스크 사용률 (%)
        'load1': (2.5, 1.0),          # 시스템 로드 (1분)
        'bytes_sent': (5000, 2000),   # 네트워크 송신 바이트
        'bytes_recv': (8000, 3000),   # 네트워크 수신 바이트
    }
    
    # 사용할 열 선택
    if cols is None:
        use_cols = default_cols
    else:
        use_cols = {col: default_cols.get(col, (500, 100)) for col in cols if col in default_cols}
        # 지정된 열이 기본 열에 없으면 임의 생성
        for col in cols:
            if col not in use_cols:
                use_cols[col] = (500, 100)  # 기본값
    
    # 데이터 생성
    data = {'timestamp': timestamps}
    for col, (mean, std) in use_cols.items():
        data[col] = np.random.normal(mean, std, size=len(timestamps))
    
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    # 시간에 따른 트렌드 추가
    days_passed = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / (24 * 3600)
    
    # 각 열에 특성 추가
    if 'usage_user' in use_cols:
        # CPU 사용률 트렌드 (주간 패턴)
        df['usage_user'] += 15 * np.sin(days_passed * 2 * np.pi / 7)  # 7일 주기
    
    if 'temp_input' in use_cols:
        # 온도 트렌드 (일간 패턴)
        hour_of_day = df['timestamp'].dt.hour
        df['temp_input'] += 5 * np.sin(hour_of_day * 2 * np.pi / 24)  # 24시간 주기
    
    if 'disk_used_percent' in use_cols:
        # 디스크 사용률 (점진적 증가)
        df['disk_used_percent'] += days_passed * 0.2
    
    # 이상치 추가 (5% 비율로 무작위 추가)
    for col in use_cols.keys():
        anomaly_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
        _, std = use_cols[col]
        df.loc[anomaly_indices, col] += np.random.uniform(1.5 * std, 3 * std, size=len(anomaly_indices))
    
    # 특정 시간에 모든 센서 값이 크게 변하는 이벤트 추가
    event_start = np.random.randint(0, int(len(df) * 0.7))
    event_duration = 24  # 24시간 지속
    if event_start + event_duration < len(df):
        for col in use_cols.keys():
            _, std = use_cols[col]
            if col == 'usage_user' or col == 'usage_system':
                df.loc[event_start:event_start + event_duration, col] *= 1.5
            elif col == 'temp_input':
                df.loc[event_start:event_start + event_duration, col] += 8
            elif col == 'usage_idle':
                df.loc[event_start:event_start + event_duration, col] *= 0.7
            else:
                # 기타 센서는 임의 변동
                df.loc[event_start:event_start + event_duration, col] *= np.random.uniform(0.8, 1.3)
    
    # 타임스탬프를 인덱스로 설정
    df.set_index('timestamp', inplace=True)
    
    # 데이터 범위 조정 (음수 값 제거 등)
    for col in use_cols.keys():
        if col == 'usage_user' or col == 'usage_system' or col == 'usage_idle' or col == 'usage_iowait':
            df[col] = df[col].clip(0, 100)
        elif col == 'mem_used_percent' or col == 'disk_used_percent':
            df[col] = df[col].clip(0, 100)
        elif col == 'temp_input':
            df[col] = df[col].clip(10, 90)
        elif col == 'load1':
            df[col] = df[col].clip(0, None)
        else:
            # 기타 센서는 음수 방지
            df[col] = df[col].clip(0, None)
    
    logger.info(f"테스트 데이터 생성 완료: {len(df)}개 행, {df.shape[1]}개 열")
    return df

def preprocess_data(df, target_col=None, min_samples=24):
    """
    모델 학습을 위한 데이터 전처리
    
    Args:
        df (pd.DataFrame): 입력 데이터프레임
        target_col (str): 타겟 열 이름
        min_samples (int): 최소 필요 샘플 수
        
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    if df.empty:
        logger.warning("전처리할 데이터가 비어 있습니다.")
        return df
    
    # 결측치 처리
    if df.isna().sum().sum() > 0:
        df = df.interpolate(method='time')
        logger.info(f"결측치를 보간했습니다. (전체 {df.isna().sum().sum()}개)")
    
    # 데이터 충분성 확인
    if len(df) < min_samples:
        logger.warning(f"데이터 샘플이 부족합니다. (현재 {len(df)}개, 최소 {min_samples}개 필요)")
    
    # 타깃 열 확인
    if target_col is not None and target_col not in df.columns:
        logger.warning(f"타겟 열 '{target_col}'이 데이터에 없습니다.")
        
        # 가장 적합한 대체 열 찾기
        possible_targets = ['temp_input', 'usage_user', 'usage_system', 'load1']
        for col in possible_targets:
            if col in df.columns:
                logger.info(f"대체 타겟 열로 '{col}'을 사용합니다.")
                target_col = col
                break
    
    # 이상치 처리 (표준 편차 기반)
    for col in df.columns:
        mean, std = df[col].mean(), df[col].std()
        lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
        
        # 극단적 이상치만 처리
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        if outliers.sum() > 0:
            logger.info(f"'{col}' 열에서 {outliers.sum()}개의 이상치를 처리했습니다.")
            df.loc[outliers, col] = df[col].median()  # 중앙값으로 대체
    
    return df

def adjust_config_for_data_size(config, df=None):
    """
    데이터 크기에 맞게 설정 조정
    
    Args:
        config (dict): 설정 정보
        df (pd.DataFrame): 현재 데이터프레임 (없으면 None)
        
    Returns:
        dict: 조정된 설정
    """
    adjusted_config = config.copy()
    
    # 데이터가 제공된 경우
    if df is not None and not df.empty:
        data_length = len(df)
        
        # 고장 예측 설정 조정
        failure_cfg = adjusted_config["failure_prediction"]
        if data_length < failure_cfg["input_window"]:
            # 데이터 길이에 맞게 input_window 조정 (최소 1, 최대는 데이터 길이 - 1)
            new_window = max(1, min(data_length - 1, failure_cfg["input_window"]))
            logger.warning(f"고장 예측 input_window를 {failure_cfg['input_window']}에서 {new_window}로 조정합니다.")
            failure_cfg["input_window"] = new_window
        
        # 자원 사용량 분석 설정 조정
        resource_cfg = adjusted_config["resource_analysis"]
        if data_length < resource_cfg["input_window"]:
            # 데이터 길이에 맞게 input_window 조정 (최소 1, 최대는 데이터 길이 - 24)
            pred_horizon = 24  # 기본 예측 기간
            new_window = max(1, min(data_length - pred_horizon, resource_cfg["input_window"]))
            if new_window <= 0:
                new_window = 1  # 최소 1로 설정
            logger.warning(f"자원 사용량 분석 input_window를 {resource_cfg['input_window']}에서 {new_window}로 조정합니다.")
            resource_cfg["input_window"] = new_window
    else:
        # 데이터가 없는 경우, 기본값 설정
        adjusted_config["failure_prediction"]["input_window"] = min(24, adjusted_config["failure_prediction"]["input_window"])
        adjusted_config["resource_analysis"]["input_window"] = min(48, adjusted_config["resource_analysis"]["input_window"])
        logger.warning("데이터가 없어 input_window를 기본값으로 조정합니다.")
    
    return adjusted_config