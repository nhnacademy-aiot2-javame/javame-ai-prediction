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
ENV INFLUXDB_URL=http://localhost:10288
ENV INFLUXDB_TOKEN=g-W7W0j9AE4coriQfnhHGMDnDhTZGok8bgY1NnZ6Z0EnTOsFY3SWAqDTC5fYlQ9mYnbK_doR074-a4Dgck2AOQ==
ENV INFLUXDB_ORG=javame
ENV INFLUXDB_BUCKET=aiot
ENV MYSQL_HOST=s4.java21.net
ENV MYSQL_PORT=18080
ENV MYSQL_USER=aiot02_team3
ENV MYSQL_PASSWORD=ryL7LcSp@Yiz[bR7
ENV MYSQL_DATABASE=aiot02_team3
ENV PREDICTION_INTERVAL=6
ENV LOG_LEVEL=INFO

# 로그 및 모델 저장 디렉토리 생성
RUN mkdir -p logs model_weights prediction_results

# 실행 명령
CMD ["python", "batch_runner.py"]