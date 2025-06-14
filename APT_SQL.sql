USE apt;

-- Bảng users
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('user', 'admin') NOT NULL DEFAULT 'user',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Bảng uploads
CREATE TABLE uploads (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    filename VARCHAR(255),
    file_path VARCHAR(500),
    user_id BIGINT NOT NULL,
    upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
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

-- Bảng stage_summary
CREATE TABLE stage_summary (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    upload_id BIGINT,
    stage_label VARCHAR(100),
    count INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE,
    UNIQUE (upload_id, stage_label)
);

INSERT INTO users (username, email, password_hash, role, created_at, updated_at)
VALUES (
    'admin',
    'admin@gmail.com',
    '$2b$12$dCDH4YE76PH8Z8AVsUy7RuCTouyswJVtlSJ7XgzpVzWVe4ipONzhW', 
    'admin',
    NOW(),
    NOW()
);

SELECT * FROM users;

