#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InfluxDB CSV 파일의 구조를 간단하게 분석하는 스크립트
pandas 버전 호환성을 고려한 버전
"""

import os
import pandas as pd
import json
from datetime import datetime

# CSV 파일 목록 - data 폴더 내부
csv_files = [
    'data/cpu_usageUser.csv',
    'data/cpu_usageSystem.csv',
    'data/cpu_usageIowait.csv',
    'data/cpu_usageIdle.csv',
    'data/mem_usedPercent.csv', 
    'data/mem_availablePercent.csv',
    'data/disk_usedPercent.csv',
    'data/diskio_data.csv',
    'data/sensors_tempInput.csv',
    'data/system_data.csv'
]

def analyze_influxdb_csv(file_path):
    """InfluxDB CSV 파일의 구조 분석"""
    print(f"\n{'='*80}")
    print(f"파일: {file_path}")
    print(f"{'='*80}")
    
    try:
        # 파일 존재 및 크기 확인
        if not os.path.exists(file_path):
            print(f"파일이 존재하지 않습니다: {file_path}")
            return None
            
        file_size = os.path.getsize(file_path)
        print(f"파일 크기: {file_size/1024:.2f} KB")
        
        # 원본 파일의 첫 몇 줄 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = []
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
        
        print("\n원본 파일의 첫 10줄:")
        for i, line in enumerate(lines, 1):
            print(f"{i}: {line[:150]}..." if len(line) > 150 else f"{i}: {line}")
        
        # pandas로 CSV 읽기 (단순화된 방식)
        try:
            df = pd.read_csv(file_path, skiprows=3)  # 처음 3행(헤더 메타데이터) 건너뛰기
            print(f"\n데이터프레임 정보:")
            print(f"- 행 수: {len(df)}")
            print(f"- 열 수: {len(df.columns)}")
            print(f"- 컬럼: {df.columns.tolist()}")
            
            # 컬럼 내용 분석
            if '_measurement' in df.columns:
                measurements = df['_measurement'].unique()
                print(f"\n측정 타입: {measurements.tolist()}")
            
            if '_field' in df.columns:
                fields = df['_field'].unique()
                print(f"필드: {fields.tolist()}")
            
            if '_value' in df.columns:
                # 일부 값을 수치형으로 변환 시도
                numeric_values = pd.to_numeric(df['_value'], errors='ignore')
                print(f"\n값 범위: {numeric_values.min()} ~ {numeric_values.max()}")
            
            # 샘플 데이터 출력
            print("\n샘플 데이터 (상위 3개 행):")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(df.head(3))
            
            return {
                'file': os.path.basename(file_path),
                'measurements': df['_measurement'].unique().tolist() if '_measurement' in df.columns else [],
                'fields': df['_field'].unique().tolist() if '_field' in df.columns else [],
                'columns': df.columns.tolist(),
                'rows': len(df)
            }
            
        except Exception as e:
            print(f"pandas 읽기 오류: {e}")
            # 대체 방법: 직접 파싱
            headers = lines[3].split(',') if len(lines) > 3 else []
            data_lines = lines[4:] if len(lines) > 4 else []
            
            print(f"\n수동 파싱 결과:")
            print(f"헤더: {headers}")
            if data_lines:
                print(f"첫 데이터 행: {data_lines[0]}")
            
            return None
        
    except Exception as e:
        print(f"파일 분석 중 오류 발생: {e}")
        return None

def main():
    """메인 함수"""
    print("InfluxDB CSV 파일 분석")
    print(f"분석 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # data 폴더 확인
    if not os.path.exists('data'):
        print("data 폴더가 존재하지 않습니다.")
        return
    
    # 각 CSV 파일 분석
    summary = {}
    for csv_file in csv_files:
        result = analyze_influxdb_csv(csv_file)
        if result:
            summary[result['file']] = result
    
    # 요약 출력
    print(f"\n{'='*80}")
    print("데이터 구조 요약")
    print(f"{'='*80}")
    
    if summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
        # 모든 측정 타입과 필드 수집
        all_measurements = set()
        all_fields = set()
        for info in summary.values():
            all_measurements.update(info['measurements'])
            all_fields.update(info['fields'])
        
        print(f"\n전체 측정 타입: {list(all_measurements)}")
        print(f"전체 필드: {list(all_fields)}")
    else:
        print("분석된 데이터가 없습니다.")
    
    # 설정 파일과 비교
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        print(f"\n{'='*80}")
        print("설정 파일과의 비교")
        print(f"{'='*80}")
        
        config_measurement = config['influxdb']['measurement']
        print(f"설정의 measurement: {config_measurement}")
        if config_measurement not in all_measurements:
            print(f"⚠️ 주의: 설정의 measurement '{config_measurement}'가 실제 데이터에 없습니다!")
        
        target_column = config['failure_prediction']['target_column']
        print(f"\n설정의 타겟 컬럼: {target_column}")
        if target_column not in all_fields:
            print(f"⚠️ 주의: 타겟 컬럼 '{target_column}'이 실제 데이터에 없습니다!")
            
    except Exception as e:
        print(f"설정 파일 비교 중 오류: {e}")

if __name__ == "__main__":
    main()