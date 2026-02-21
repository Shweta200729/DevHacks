CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100)        NOT NULL,
    email           VARCHAR(255)        NOT NULL UNIQUE,
    phone           VARCHAR(15)         NOT NULL,
    password_hash   VARCHAR(255)        NOT NULL,
    created_at      TIMESTAMP           DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP           DEFAULT CURRENT_TIMESTAMP
);