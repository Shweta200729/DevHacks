# Technical Stack

This document outlines the end-to-end technology stack powering the Decentralized Federated Learning Platform. The stack was chosen to maximize performance (ML processing), developer velocity, and visual impact.

---

## 1. Backend / ML Aggregator (Python)

The backend acts as the Orchestrator, the Aggregator, and the Security Checkpoint.

*   **FastAPI:** The core web framework. Chosen for its native asynchronous support (`asyncio`), high performance, and auto-generated Swagger documentation. It handles the REST endpoints that the frontend consumes.
*   **PyTorch (`torch`, `torchvision`):** The beating heart of the Machine Learning engine. 
    *   Handles dataset loading (`DataLoader`).
    *   Dynamically builds Neural Networks (`model_factory.py`) depending on the shape of the uploaded data (CNNs for 3D images, MLPs for 1D tabular data).
    *   Executes local training loops (`loss.backward()`, `optimizer.step()`).
*   **Pandas:** Used for parsing, cleaning, and validating user-uploaded CSV datasets before they are converted into PyTorch tensors.
*   **Pydantic:** Strictly validates incoming JSON payloads (e.g., `SimulationRequest`, `AttackProbeRequest`) to ensure the server doesn't crash from malformed frontend requests.
*   **Uvicorn:** The ASGI web server used to run the FastAPI application in development and production.

---

## 2. Frontend / Client Dashboard (TypeScript)

The frontend simulates the Client/Edge Nodes and provides the command center for the network administrator.

*   **Next.js (App Router):** The React framework used for full-stack capabilities, server-side rendering (SSR), and optimized routing. Chosen for its robust architecture and SEO/performance defaults.
*   **React:** The core UI library. Heavily utilizes hooks (`useState`, `useEffect`, `useRef`) to manage the complex, real-time state of the simulations and attack gauges.
*   **TypeScript:** Enforces strict type safety across the application, preventing runtime bugs and ensuring the UI perfectly maps to the API payloads expected by FastAPI.
*   **Tailwind CSS:** Utility-first CSS framework. Used to rapidly build the "premium, stunning" UI, handling responsive design, dark mode, complex gradients, and glassmorphic effects.
*   **Lucide React:** The iconography library. Provides lightweight, clean, and consistent SVG icons across the dashboard.
*   **Recharts:** A composable charting library built on React components. Used to render the real-time "Accuracy vs. Loss" line charts and network health graphs.
*   **Sonner:** A toast notification library used to provide instant user feedback (e.g., "Attack Successfully Caught!", "Dataset Uploaded").

---

## 3. Infrastructure & Database (Supabase)

Supabase serves as the open-source Firebase alternative, providing PostgreSQL database management and blob storage.

*   **Supabase Database (PostgreSQL):** 
    *   Stores the transactional ledger of the system.
    *   Manages users, wallet balances (`wallets` table).
    *   Logs system events, training history, and accuracy/loss metrics over time (`evaluation_metrics` table).
*   **Supabase Storage:** 
    *   Acts as the distributed file system.
    *   Stores the heavy artifacts securely: the raw `.csv` datasets and the serialized `.pt` (PyTorch) global model configuration files (`global_model_v1.pt`, `v2.pt`, etc.).
*   **Supabase Client SDKs:** Used in both the Python backend and the Next.js frontend to interact seamlessly with the database and storage buckets.

---

## 4. Architecture Flow

1.  **Frontend** sends `POST` request to **FastAPI**.
2.  **FastAPI** spins up a **PyTorch** async background worker to avoid blocking the API.
3.  **PyTorch worker** pulls the `.pt` model from **Supabase Storage**.
4.  Worker trains locally, applies attack vectors if configured, and calculates gradient variations.
5.  Worker sends gradients to the `detection.py` engine for L2 Norm evaluation.
6.  If passed, the model is aggregated (`FedAvg`) and the new `.pt` file is pushed to **Supabase Storage**.
7.  The `blockchain.py` simulator writes a token reward/slash transaction to the **Supabase Database**.
8.  **Frontend** polls the DB and dynamically updates the UI using **Recharts**.
