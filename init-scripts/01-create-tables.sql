-- 예측 결과 테이블 생성

-- 1. 고장 예측 결과 테이블
CREATE TABLE IF NOT EXISTS failure_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    device_id VARCHAR(50),
    failure_probability FLOAT,
    is_failure_predicted BOOLEAN,
    prediction_time DATETIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp),
    INDEX idx_device_id (device_id)
);

-- 2. 자원 사용량 예측 결과 테이블
CREATE TABLE IF NOT EXISTS resource_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    device_id VARCHAR(50),
    resource_type VARCHAR(50),
    predicted_value FLOAT,
    prediction_time DATETIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp),
    INDEX idx_device_id (device_id),
    INDEX idx_resource_type (resource_type)
);

-- 3. 용량 증설 계획 테이블
CREATE TABLE IF NOT EXISTS capacity_planning (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scenario VARCHAR(50),
    resource_type VARCHAR(50),
    expansion_date DATE,
    threshold_value FLOAT,
    prediction_time DATETIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_scenario (scenario),
    INDEX idx_resource_type (resource_type)
);

-- 4. 예측 실행 로그 테이블
CREATE TABLE IF NOT EXISTS prediction_runs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_mode VARCHAR(20),
    status VARCHAR(20),
    start_time DATETIME,
    end_time DATETIME,
    execution_time_sec INT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_run_mode (run_mode),
    INDEX idx_status (status),
    INDEX idx_start_time (start_time)
);