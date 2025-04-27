# Dockerfile 수정
FROM python:3.9-slim

# 호스트와 동일한 UID/GID를 가진 사용자 생성
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

WORKDIR /app

# 필요한 패키지 설치 및 앱 복사
COPY --chown=appuser:appuser . .

# 사용자 전환
USER appuser

CMD ["python", "batch_runner.py"]