# data_exploration.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta

# 설정 파일 로드
with open('config.json', 'r') as f:
    config = json.load(f)

influx_config = config["influxdb"]

# 클라이언트 생성
client = InfluxDBClient(
    url=influx_config["url"],
    token=influx_config["token"],
    org=influx_config["org"]
)

# 최근 데이터 기간 설정
start_time = datetime.now() - timedelta(days=7)  # 최근 7일 데이터
start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')

# 데이터 로드 함수
def query_data_by_origin(origin, bucket=influx_config["bucket"]):
    """origin 태그별 데이터 쿼리"""
    query_api = client.query_api()
    
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: {start_time_str})
      |> filter(fn: (r) => r["origin"] == "{origin}")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    result = query_api.query_data_frame(query)
    
    if result.empty:
        print(f"'{origin}' origin에서 데이터를 찾을 수 없습니다.")
        return pd.DataFrame()
    
    # 중복된 컬럼 처리
    if isinstance(result, list):
        result = pd.concat(result)
    
    # _time을 인덱스로 설정
    if '_time' in result.columns:
        result.set_index('_time', inplace=True)
    
    # 필요한 컬럼만 선택
    exclude_cols = ['result', 'table', '_start', '_stop', '_measurement']
    data_cols = [col for col in result.columns if col not in exclude_cols]
    
    return result[data_cols]

# 각 origin별 데이터 로드
server_data = query_data_by_origin("server_data")
sensor_data = query_data_by_origin("sensor_data")

print("\n서버 데이터 정보:")
print(f"행 수: {len(server_data)}")
print(f"열 수: {server_data.shape[1]}")
print(f"컬럼: {server_data.columns.tolist()}")
print("\n기본 통계량:")
print(server_data.describe())

print("\n센서 데이터 정보:")
print(f"행 수: {len(sensor_data)}")
print(f"열 수: {sensor_data.shape[1]}")
print(f"컬럼: {sensor_data.columns.tolist()}")
print("\n기본 통계량:")
print(sensor_data.describe())

# 데이터 시각화
plt.figure(figsize=(15, 8))

# 서버 CPU 사용량
if 'usage_user' in server_data.columns:
    plt.subplot(2, 2, 1)
    plt.plot(server_data.index, server_data['usage_user'])
    plt.title('CPU 사용자 사용률')
    plt.ylabel('사용률 (%)')
    plt.grid(True)

# 서버 메모리 가용률
if 'available_percent' in server_data.columns:
    plt.subplot(2, 2, 2)
    plt.plot(server_data.index, server_data['available_percent'])
    plt.title('메모리 가용률')
    plt.ylabel('가용률 (%)')
    plt.grid(True)

# 온도 데이터
if 'temperature' in sensor_data.columns:
    plt.subplot(2, 2, 3)
    plt.plot(sensor_data.index, sensor_data['temperature'])
    plt.title('서버룸 온도')
    plt.ylabel('온도 (°C)')
    plt.grid(True)

# 습도 데이터
if 'humidity' in sensor_data.columns:
    plt.subplot(2, 2, 4)
    plt.plot(sensor_data.index, sensor_data['humidity'])
    plt.title('서버룸 습도')
    plt.ylabel('습도 (%)')
    plt.grid(True)

plt.tight_layout()
plt.savefig('data_exploration.png')
plt.close()

# 데이터 병합 테스트
print("\n데이터 병합 테스트:")
# 두 데이터 모두 시간 인덱스를 가지고 있으므로 외부 조인으로 병합
merged_data = pd.merge(server_data, sensor_data, 
                      left_index=True, right_index=True, 
                      how='outer',
                      suffixes=('_server', '_sensor'))

print(f"병합 결과 행 수: {len(merged_data)}")
print(f"병합 결과 열 수: {merged_data.shape[1]}")

# 상관관계 분석
plt.figure(figsize=(12, 10))
numeric_cols = merged_data.select_dtypes(include=['number']).columns
corr = merged_data[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('변수 간 상관관계')
plt.tight_layout()
plt.savefig('correlation_analysis.png')

print("\n탐색 완료! 결과 이미지 저장됨: data_exploration.png, correlation_analysis.png")
client.close()