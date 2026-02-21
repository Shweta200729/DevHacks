-- Run this inside your Supabase SQL Editor to create backend tables

CREATE TABLE clients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version_num INT UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE client_updates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version_id INT REFERENCES model_versions(id) ON DELETE CASCADE,
    client_id UUID REFERENCES clients(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('ACCEPT', 'REJECT')),
    norm_value NUMERIC,
    distance_value NUMERIC,
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE evaluation_metrics (
    id SERIAL PRIMARY KEY,
    version_id INT REFERENCES model_versions(id) ON DELETE CASCADE,
    loss NUMERIC NOT NULL,
    accuracy NUMERIC NOT NULL,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE aggregation_logs (
    id SERIAL PRIMARY KEY,
    version_id INT REFERENCES model_versions(id) ON DELETE CASCADE,
    total_accepted INT,
    total_rejected INT,
    method TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert a baseline model to kick off Round 0
INSERT INTO model_versions (version_num, file_path) VALUES (0, 'models/global_model_round_0.pt');
