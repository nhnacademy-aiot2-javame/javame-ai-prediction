FROM python:3.9-slim

WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치를 위한 requirements.txt 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 시간대 설정
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 환경 변수 설정 (기본값)
ENV INFLUXDB_URL=http://influxdb:8086
ENV INFLUXDB_TOKEN=your_token
ENV INFLUXDB_ORG=your_org
ENV INFLUXDB_BUCKET=your_bucket
ENV MYSQL_HOST=mysql
ENV MYSQL_PORT=3306
ENV MYSQL_USER=user
ENV MYSQL_PASSWORD=password
ENV MYSQL_DATABASE=predictions
ENV PREDICTION_INTERVAL=6
ENV LOG_LEVEL=INFO

# 로그 및 모델 저장 디렉토리 생성
RUN mkdir -p logs model_weights prediction_results

# 실행 명령
CMD ["python", "batch_runner.py"]