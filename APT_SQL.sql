USE apt;

-- Bảng uploads (không còn user_id)
CREATE TABLE uploads (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    filename VARCHAR(255),
    file_path VARCHAR(500),
    upload_time DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Bảng predictions
CREATE TABLE predictions (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    upload_id BIGINT,
    log_data TEXT,
    predicted_label VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
);

-- Bảng risk_summary
CREATE TABLE risk_summary (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    upload_id BIGINT,
    risk_level VARCHAR(100),
    count INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE,
    UNIQUE (upload_id, risk_level)
);
