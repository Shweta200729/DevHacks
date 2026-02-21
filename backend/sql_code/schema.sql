-- Full Backend Supabase Database Schema
-- Includes Web Application Users and Federated Learning Orchestration Models

-- 1. Create extension for UUID generation (if using UUIDs)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ----------------------------------------------------
-- A. User Authentication (Dashboard Application)
-- ----------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    phone VARCHAR(15) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ----------------------------------------------------
-- B. Federated Learning Actors & Artifacts
-- ----------------------------------------------------

-- Step 1: Federated Learning Clients Table
-- Using TEXT to allow custom client IDs sent from edge devices or Python simulation scripts
CREATE TABLE IF NOT EXISTS clients (
    id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::TEXT,
    client_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Step 2: Global Model Versions Tracking
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version_num INT UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ----------------------------------------------------
-- C. Federated Learning Aggregation Logs
-- ----------------------------------------------------

-- Step 3: Logging Individual Client Model Updates
CREATE TABLE IF NOT EXISTS client_updates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version_id INT REFERENCES model_versions(id) ON DELETE CASCADE,
    client_id TEXT REFERENCES clients(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('ACCEPT', 'REJECT')),
    norm_value NUMERIC,
    distance_value NUMERIC,
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Step 4: Tracking Global Evaluation Metrics
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id SERIAL PRIMARY KEY,
    version_id INT REFERENCES model_versions(id) ON DELETE CASCADE,
    loss NUMERIC NOT NULL,
    accuracy NUMERIC NOT NULL,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Step 5: Master Aggregation Execution Logs
CREATE TABLE IF NOT EXISTS aggregation_logs (
    id SERIAL PRIMARY KEY,
    version_id INT REFERENCES model_versions(id) ON DELETE CASCADE,
    total_accepted INT,
    total_rejected INT,
    method TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ----------------------------------------------------
-- D. Initial Data Seeding
-- ----------------------------------------------------
-- Insert a baseline model to kick off Round 0
INSERT INTO model_versions (version_num, file_path) 
VALUES (0, 'models/global_model_round_0.pt')
ON CONFLICT (version_num) DO NOTHING;
