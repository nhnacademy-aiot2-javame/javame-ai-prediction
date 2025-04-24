#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
시각화 유틸리티

데이터 및 모델 결과의 시각화를 위한 함수 제공
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# 한글 폰트 설정 (나눔 폰트가 있는 경우)
try:
    import matplotlib.font_manager as fm
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 나눔고딕 폰트 경로
    
    if os.path.exists(font_path):
        # 폰트 설정
        plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False
        logger.info("한글 폰트(나눔고딕) 설정 완료")
    else:
        # 기본 폰트 사용
        logger.warning("나눔고딕 폰트 파일을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
except Exception as e:
    logger.warning(f"폰트 설정 중 오류 발생: {e}")

def ensure_dir(directory):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"디렉토리 생성: {directory}")

def plot_sensor_data(df, cols=None, save_path=None, figsize=(15, 10)):
    """
    센서 데이터 시각화
    
    Args:
        df (pd.DataFrame): 시각화할 데이터프레임
        cols (list): 시각화할 열 이름 목록 (None이면 전체 열)
        save_path (str): 저장 경로 (None이면 표시만 함)
        figsize (tuple): 그림 크기
    """
    if df.empty:
        logger.warning("시각화할 데이터가 비어 있습니다.")
        return
    
    # 시각화할 열 선택
    if cols is None:
        cols = df.columns.tolist()
    else:
        # 존재하는 열만 필터링
        cols = [col for col in cols if col in df.columns]
    
    # cols가 비어있는지 확인 (리스트 형태로 확인)
    if not cols:
        logger.warning("시각화할 열이 없습니다.")
        return
    
    # 서브플롯 개수 계산
    n_plots = len(cols)
    n_rows = (n_plots + 1) // 2  # 2열로 배치
    
    plt.figure(figsize=figsize)
    
    # 각 센서 데이터 플롯
    for i, col in enumerate(cols):
        plt.subplot(n_rows, 2, i+1)
        plt.plot(df.index, df[col], label=col)
        plt.title(f'{col} 데이터')
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    # 결과 저장 또는 표시
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"센서 데이터 시각화 저장: {save_path}")
    else:
        plt.show()

def plot_failure_prediction(df, target_col, results, threshold=0.5, save_path=None, figsize=(15, 10)):
    """
    고장 예측 결과 시각화
    
    Args:
        df (pd.DataFrame): 원본 데이터
        target_col (str): 예측한 타겟 열
        results (pd.DataFrame): 예측 결과
        threshold (float): 고장 임계값
        save_path (str): 저장 경로 (None이면 표시만 함)
        figsize (tuple): 그림 크기
    """
    # 결과가 비어있는지 확인
    if results.empty:
        logger.warning("시각화를 위한 결과가 비어 있습니다.")
        return
    
    # 타겟 열이 원본 데이터에 있는지 확인
    if target_col not in df.columns:
        logger.warning(f"타겟 열 '{target_col}'이 원본 데이터에 없습니다.")
        return
    
    # 시각화를 위해 결과를 인덱싱
    if 'timestamp' in results.columns:
        results.set_index('timestamp', inplace=True)
    
    plt.figure(figsize=figsize)
    
    # 고장 확률 플롯
    plt.subplot(2, 1, 1)
    plt.plot(results.index, results['failure_probability'], label='고장 확률', color='blue')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'고장 임계값 ({threshold})')
    
    # 고장 지점 강조
    failure_idx = results[results['is_failure_predicted'] == True].index
    if len(failure_idx) > 0:
        plt.scatter(failure_idx, results.loc[failure_idx, 'failure_probability'], color='red', label='예측된 고장')
            
    plt.title('고장 확률 추이')
    plt.ylabel('고장 확률')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 원본 데이터와 고장 지점
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df[target_col], label=f'원본 {target_col}', alpha=0.7)
    
    # 고장 지점 강조
    if len(failure_idx) > 0:
        for idx in failure_idx:
            if idx in df.index:
                plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
            
    plt.title(f'{target_col} 원본 데이터 및 고장 예측')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 결과 저장 또는 표시
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"고장 예측 시각화 저장: {save_path}")
    else:
        plt.show()

def plot_resource_prediction(df, target_col, results, save_path=None, figsize=(15, 8)):
    """
    자원 사용량 예측 결과 시각화
    
    Args:
        df (pd.DataFrame): 원본 데이터
        target_col (str): 예측한 타겟 열
        results (pd.DataFrame): 예측 결과
        save_path (str): 저장 경로 (None이면 표시만 함)
        figsize (tuple): 그림 크기
    """
    # 결과가 비어있는지 확인
    if results.empty:
        logger.warning("시각화를 위한 결과가 비어 있습니다.")
        return
            
    plt.figure(figsize=figsize)
    
    # 원본 데이터와 예측 결과 플롯
    plt.plot(df.index, df[target_col], label=f'원본 {target_col}', alpha=0.7)
    plt.plot(results.index, results['predicted_value'], label='예측 값', color='red')
    
    # 미래 예측 부분 강조
    hist_prediction = results.loc[df.index.intersection(results.index)]
    future_prediction = results.loc[~results.index.isin(df.index)]
    
    if not future_prediction.empty:
        plt.plot(future_prediction.index, future_prediction['predicted_value'], 
                label='미래 예측', color='green', linestyle='--')
    
    plt.title(f'{target_col} 자원 사용량 예측')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 결과 저장 또는 표시
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"자원 사용량 예측 시각화 저장: {save_path}")
    else:
        plt.show()

def plot_long_term_scenarios(scenario_results, capacity_threshold=None, expansion_dates=None, save_path=None, figsize=(15, 10)):
    """
    장기 시나리오 예측 결과 시각화
    
    Args:
        scenario_results (dict): 시나리오별 예측 결과
        capacity_threshold (float): 용량 임계값
        expansion_dates (dict): 시나리오별 용량 증설 필요 시점
        save_path (str): 저장 경로 (None이면 표시만 함)
        figsize (tuple): 그림 크기
    """
    # 결과가 비어있는지 확인
    if not scenario_results:
        logger.warning("시각화를 위한 시나리오 결과가 비어 있습니다.")
        return
            
    plt.figure(figsize=figsize)
    
    colors = {
        '기본': 'blue',
        '저성장': 'green',
        '고성장': 'red'
    }
    
    # 각 시나리오 플롯
    for scenario_name, forecast in scenario_results.items():
        plt.plot(forecast['ds'], forecast['yhat'], 
                label=f'{scenario_name} 시나리오', 
                color=colors.get(scenario_name, 'gray'))
        
        # 예측 구간 추가
        plt.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            color=colors.get(scenario_name, 'gray'),
            alpha=0.2
        )
        
        # 용량 증설 필요 시점 표시
        if expansion_dates and scenario_name in expansion_dates and expansion_dates[scenario_name]:
            plt.axvline(
                x=expansion_dates[scenario_name],
                color=colors.get(scenario_name, 'gray'),
                linestyle='--',
                alpha=0.7,
                label=f'{scenario_name} 증설 시점'
            )
    
    # 용량 임계값 표시
    if capacity_threshold:
        plt.axhline(
            y=capacity_threshold,
            color='black',
            linestyle='-.',
            label=f'용량 임계값 ({capacity_threshold})'
        )
    
    plt.title('장기 자원 사용량 예측 시나리오')
    plt.ylabel('자원 사용량')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 결과 저장 또는 표시
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"장기 시나리오 시각화 저장: {save_path}")
    else:
        plt.show()
def plot_correlation_matrix(df, save_path=None, figsize=(12, 10)):
    """
    상관관계 행렬 시각화
    
    Args:
        df (pd.DataFrame): 데이터프레임
        save_path (str): 저장 경로 (None이면 표시만 함)
        figsize (tuple): 그림 크기
    """
    if df.empty:
        logger.warning("상관관계 분석을 위한 데이터가 비어 있습니다.")
        return
    
    # 숫자형 열만 선택
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        logger.warning("상관관계 분석을 위한 숫자형 열이 2개 이상 필요합니다.")
        return
    
    # 상관관계 계산
    corr = numeric_df.corr()
    
    # 시각화
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        annot=True, 
        fmt=".2f"
    )
    
    plt.title('센서 데이터 상관관계 행렬')
    plt.tight_layout()
    
    # 결과 저장 또는 표시
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"상관관계 행렬 시각화 저장: {save_path}")
    else:
        plt.show()

def plot_feature_importance(df, target_col, n_features=10, save_path=None, figsize=(12, 8)):
    """
    특성 중요도 시각화 (RandomForest 기반)
    
    Args:
        df (pd.DataFrame): 데이터프레임
        target_col (str): 타겟 열 이름
        n_features (int): 표시할 특성 수
        save_path (str): 저장 경로 (None이면 표시만 함)
        figsize (tuple): 그림 크기
    """
    if df.empty:
        logger.warning("특성 중요도 분석을 위한 데이터가 비어 있습니다.")
        return
    
    if target_col not in df.columns:
        logger.warning(f"타겟 열 '{target_col}'이 데이터에 없습니다.")
        return
    
    try:
        # RandomForest 모델 생성 및 학습
        from sklearn.ensemble import RandomForestRegressor
        
        # 특성과 타겟 분리
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 숫자형 특성만 선택
        X = X.select_dtypes(include=[np.number])
        
        if X.shape[1] == 0:
            logger.warning("특성 중요도 분석을 위한 숫자형 특성이 없습니다.")
            return
        
        # 모델 학습
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 특성 중요도 계산
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 상위 n개 특성 선택
        top_indices = indices[:n_features]
        top_features = X.columns[top_indices]
        top_importances = importances[top_indices]
        
        # 시각화
        plt.figure(figsize=figsize)
        plt.title(f'{target_col}에 대한 특성 중요도')
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('중요도')
        plt.ylabel('특성')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # 결과 저장 또는 표시
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300)
            plt.close()
            logger.info(f"특성 중요도 시각화 저장: {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"특성 중요도 분석 중 오류 발생: {e}")
def plot_server_environment_correlation(df, server_cols, env_cols, save_path=None, figsize=(15, 10)):
    """
    서버 성능과 환경 데이터 간의 상관관계 시각화
    
    Args:
        df (pd.DataFrame): 데이터프레임
        server_cols (list): 서버 성능 관련 열 이름 목록
        env_cols (list): 환경 관련 열 이름 목록
        save_path (str): 저장 경로 (None이면 표시만 함)
        figsize (tuple): 그림 크기
    """
    plt.figure(figsize=figsize)
    
    # 열 존재 여부 확인
    server_cols = [col for col in server_cols if col in df.columns]
    env_cols = [col for col in env_cols if col in df.columns]
    
    if not server_cols or not env_cols:
        logger.warning("서버 또는 환경 데이터 열이 없습니다.")
        return
    
    # 상관관계 분석 및 시각화
    plt.subplot(2, 1, 1)
    # 시간에 따른 추이 시각화
    for col in server_cols:
        # 정규화하여 같은 스케일로 표시
        normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        plt.plot(df.index, normalized, label=f'{col} (정규화)')
    
    for col in env_cols:
        # 정규화하여 같은 스케일로 표시
        normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        plt.plot(df.index, normalized, label=f'{col} (정규화)', linestyle='--')
    
    plt.title('서버 성능과 환경 데이터 시계열 비교')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 산점도 매트릭스
    plt.subplot(2, 1, 2)
    corr_cols = server_cols + env_cols
    corr_matrix = df[corr_cols].corr()
    
    # 히트맵으로 상관관계 표시
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('서버-환경 변수 간 상관관계')
    
    plt.tight_layout()
    
    # 결과 저장 또는 표시
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"서버-환경 상관관계 시각화 저장: {save_path}")
    else:
        plt.show()