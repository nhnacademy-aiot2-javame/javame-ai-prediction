# check_influxdb_data.py
import json
from influxdb_client import InfluxDBClient

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

# 쿼리 실행
query_api = client.query_api()
query = f'''
from(bucket: "{influx_config["bucket"]}")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement == "{influx_config["measurement"]}")
  |> limit(n: 20)
'''

# 결과 출력
tables = query_api.query(query=query)
print(f"쿼리 결과 테이블 수: {len(tables)}")

for table in tables:
    print(f"\n테이블: {table.records[0].get_measurement() if table.records else 'Empty'}")
    print("필드 | 값 | 타입")
    print("-----|-----|-----")
    count = 0
    for record in table.records:
        if(count == 1):
            break
        else:
            count += 1
        print(f"{record.get_field()} | {record.get_value()} | {type(record.get_value())}")
        
client.close()             