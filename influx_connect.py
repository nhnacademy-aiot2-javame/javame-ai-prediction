from influxdb_client import InfluxDBClient
import warnings
from influxdb_client.client.warnings import MissingPivotFunction

# 피벗 함수 경고 억제
warnings.simplefilter("ignore", MissingPivotFunction)

# 연결 매개변수
INFLUX_URL = "https://influxdb.javame.live"
INFLUX_TOKEN = "g-W7W0j9AE4coriQfnhHGMDnDhTZGok8bgY1NnZ6Z0EnTOsFY3SWAqDTC5fYlQ9mYnbK_doR074-a4Dgck2AOQ=="
INFLUX_ORG = "javame"
INFLUX_BUCKET = "data"




# 클라이언트 연결 및 오류 처리 개선
try:
    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )
    
    # 연결 상태 확인
    health = client.health()
    print(f"서버 상태: {health.status}")
    
    read_api = client.query_api()
    
    # 개선된 쿼리 (피벗 함수 추가)
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -7d)
      |> filter(fn: (r) => r["origin"] == "server_data")
      |> filter(fn: (r) => r["_field"] == "value")
      |> limit(n: 5)
      |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
      |> yield(name: "mean")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    df = read_api.query_data_frame(query)
    print("쿼리 결과:")
    print(df.head())
    
except Exception as e:
    print(f"오류 발생: {e}")
    
    # 인증 오류인 경우 추가 디버깅 정보 제공
    if "401" in str(e) or "unauthorized" in str(e).lower():
        print("\n인증 문제가 발생했습니다. 다음을 확인하세요:")
        print("1. 토큰이 올바른지 확인")
        print("2. 토큰이 만료되지 않았는지 확인")
        print("3. URL, 조직 이름, 버킷 이름이 정확한지 확인")
        print("4. HTTPS 대신 HTTP를 사용하는 경우, 서버 설정 확인")
finally:
    if 'client' in locals():
        client.close()