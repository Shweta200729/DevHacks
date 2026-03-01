# Product Requirements Document (PRD)

## 1. Product Name
**Decentralized Federated Learning Platform (with Byzantine Fault Tolerance & Tokenomics)**

## 2. Product Vision & Summary
Traditional Machine Learning relies on centralized data collection, creating massive privacy risks and data silos. This platform solves the privacy problem through **Federated Learning (FL)**: bringing the model to the data instead of bringing the data to the model. 

To ensure the ecosystem is secure and incentivized, the platform integrates **Decentralized Tokenomics** (rewarding nodes for computing power) and a **Byzantine Defense Engine** (protecting the global model from malicious actors trying to poison the data). 

The goal is to provide a complete, interactive, and visually stunning dashboard where users can simulate decentralized training, watch real-time defense mechanisms catch attackers, and see blockchain-based rewards/slashing in action.

## 3. Target Audience
*   **AI Researchers & Data Scientists:** Who want to understand or experiment with Federated Learning without setting up complex infrastructure.
*   **Enterprise Data Providers (e.g., Hospitals, Banks):** Organizations that possess sensitive data and want to contribute to ML models while preserving raw data privacy.
*   **Hackathon Judges & Investors:** Looking for a cohesive, technically deep, yet visually engaging demonstration of Web3 + AI integrations.

## 4. Key Features & Requirements

### 4.1. Core Federated Learning Engine (The "What")
*   **Requirement:** The system must train a global model collaboratively across multiple "nodes" without sharing raw data.
*   **Mechanism:** Users upload an initial dataset to provision the Global Model. The FastAPI backend orchestrates asynchronous PyTorch workers. These workers pull the global model, train it locally on their specific subset of data, and return only the calculated weight updates (gradients).
*   **Aggregation:** The backend uses the `FedAvg` (Federated Averaging) algorithm to combine validated local updates into a new, smarter global model.

### 4.2. Byzantine Fault Tolerance (The "Shield")
*   **Requirement:** The ecosystem must be robust against malicious nodes attempting to poison the model (e.g., a competitor trying to ruin the AI).
*   **Mechanism:** A dedicated `detection.py` engine intercepts every incoming weight update before aggregation.
*   **Metrics:** It calculates the `L2 Norm` (total size of the update) and the `L2 Distance` (how far the update drifts from the global model). If these exceed configured thresholds (e.g., `NORM_THRESHOLD = 1000.0`), the update is flagged as a Byzantine attack, rejected, and discarded.

### 4.3. Decentralized Tokenomics (The "Incentive")
*   **Requirement:** Users must be financially incentivized to provide honest computing power and penalized for attacks.
*   **Mechanism:** A simulated blockchain (`blockchain.py`) tracks user wallets.
*   **Rewards:** Honest nodes that submit validated updates receive a token reward (`+10 FLT`).
*   **Slashing:** Malicious nodes caught by the defense engine lose their staked tokens (`-15 FLT penalty`).

### 4.4. Interactive Attack Playground (The "Wow Factor")
*   **Requirement:** Users should be able to actively try to break the system to see the defense in action.
*   **Mechanism:** A dedicated "Attack Lab" UI where users can configure simulated attacks:
    *   **Gaussian Noise:** Blasting the weights with random static.
    *   **Label Flipping:** Corrupting the training data itself (e.g., telling the AI that pictures of cats are dogs).
    *   **Sign Flipping:** Reversing the gradient directions to make the model actively worse.
*   **Real-time Feedback:** A live gauge pings a `/attack-probe` endpoint to show exactly when an attack crosses the detection thresholds, before the user even clicks "Launch".

### 4.5. Analytics & Monitoring (The "Dashboard")
*   **Requirement:** The training progress and network health must be easily visualizable.
*   **Mechanism:** Live charts (Recharts) plotting Global Model Accuracy and Loss over time. A real-time system log displaying "Aggregated V2", "Rejected Update", "Slashed Wallet", etc.

## 5. Non-Functional Requirements
*   **Performance:** The backend must handle ML training asynchronously (`asyncio` + PyTorch background tasks) so the FastApi server never blocks or times out HTTP requests.
*   **UI/UX:** The interface must be "stunning", using a premium dark-mode aesthetic with gradients, glassmorphism, and responsive layouts.
*   **Scalability:** The architecture (FastAPI + Supabase) must be logically separable so compute nodes can eventually be physically separated from the orchestrator.

## 6. Future Roadmap
*   Deploy actual Smart Contracts on an L2 (e.g., Arbitrum or Polygon) instead of simulating the ledger.
*   Integrate Zero-Knowledge Proofs (ZK-ML) to mathematically prove the local training was executed correctly without revealing the data.
*   Support for custom user-uploaded PyTorch Architectures rather than dynamically generating standard CNNs/MLPs.
