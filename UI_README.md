# UI & User Guide README

Welcome to the Decentralized Federated Learning Dashboard. This guide will help you navigate the UI, understand the visual components, and effectively demo the platform's core features (Federated Learning, Byzantine Defense, and Tokenomics).

---

## 1. Global Navigation (The Sidebar)

The sidebar on the left is your main navigation hub. It contains links to the four core sections of the platform. By navigating through these in order, you can demonstrate the complete lifecycle of the platform.

### Navigation Items:
*   **Overview (Home):** The high-level command center. Tracks global metrics and allows you to initialize the network.
*   **Clients Node Simulator (CPU Icon):** The interface for simulating individual user devices contributing data to the network.
*   **Attack Lab (Swords Icon):** The interactive playground for acting as a malicious hacker trying to poison the AI.
*   **Training Logs (Scroll Icon):** The raw technical ledger tracking every API call, training round, and defense algorithm result.

---

## 2. Page Breakdown & How to Demo

### Step 1: The Overview Page (`/dashboard`)
**Purpose:** Initialize the system and view the health of the Global AI Model.

**Key UI Components:**
1.  **Metric Cards (Top Row):** Displays global stats: Total Clients, Total Training Rounds, Current Global Accuracy, and Global Loss.
2.  **Accuracy vs Loss Chart (Recharts):** A beautiful, interactive line chart. You can hover over data points to see historical model improvements over successive training rounds.
3.  **"Upload Local Dataset" Panel:** 
    *   **How to use:** This is the *first action* you must take. Click "Choose File" and upload a `.csv` dataset.
    *   **What it does:** It provisions the initial PyTorch model in the backend, setting the input/output layers based on your data, and saving `global_model_v1.pt` to Supabase. This makes the entire rest of the app functional.

### Step 2: The Clients Node Simulator (`/dashboard/clients`)
**Purpose:** Simulate decentralized edge devices training the model locally and earning token rewards.

**Key UI Components:**
1.  **Simulation Controller (Left Panel):**
    *   Shows the currently logged-in user's Session ID.
    *   **"Fire Normal Update" Button:** Click this to simulate an honest node. It runs a local PyTorch training loop and submits clean gradients.
2.  **Connected Edge Nodes (Right Panel):**
    *   A list of simulated devices representing the decentralized network.
3.  **Recent P2P Traffic (Bottom Terminal):**
    *   A simulated terminal window showing the raw networking logs. When you click "Fire Normal Update," you will see green logs indicating the update was evaluated and accepted.

### Step 3: The Attack Lab (`/dashboard/attack-playground`)
**Purpose:** The "Wow Factor" demo. Play the role of a malicious actor trying to destroy the AI, and watch the Byzantine Defense Engine stop you.

**Key UI Components:**
1.  **Configure Attack (Left Panel):**
    *   **Attack Type Dropdown:** Choose between `Gaussian Noise` (random data corruption), `Label Flipping` (teaching the AI wrong answers), or `Sign Flipping` (reversing the AI's learning direction).
    *   **Intensity Slider:** Drag this from "Safe" to "Extreme".
    *   **Launch Attack Button:** Commits the malicious payload to the real PyTorch backend.
2.  **Live Byzantine Detection Gauge (Top Right):**
    *   *This is interactive!* As you drag the intensity slider, these bars update in real-time (via the `/attack-probe` API) without waiting for a training loop.
    *   **L2 Norm Bar:** Shows the mathematical size of your attack. Watch it turn red when it crosses the `1000` threshold.
    *   **L2 Distance Bar:** Shows how far your poisoned update will pull the global model. Watch it turn red when it crosses the `500` threshold.
    *   Displays a real-time verdict: `WOULD BE CAUGHT` or `WOULD PASS`.
3.  **Blockchain Impact (Middle Right):**
    *   Shows the Tokenomics in action. 
    *   Watch your wallet balance drop (e.g., `-15 FLT SLASHED`) when you launch an attack and get caught.
4.  **Attack History (Bottom Right):**
    *   A scrollable log of all the attacks you've launched, their config, and the backend's verdict.

### Step 4: The Training Logs (`/dashboard/logs`)
**Purpose:** Deep technical validation for developers and judges.

**Key UI Components:**
1.  **Real-time Event Stream:** A raw, unfiltered list of every event happening in the backend (Model updates, Client slashes, Evaluation metrics).
2.  **Status Badges:** Color-coded badges (`SUCCESS` green, `REJECTED` red, `INFO` blue) make it easy to scan the ledger and prove the system is processing asynchronous ML tasks correctly.

---

## 3. Design Principles (Theme & Aesthetics)
*   **Color Palette:** The UI utilizes a premium "Dark Mode" aesthetic, relying on deep indigos, subtle violets, and stark white text for high contrast.
*   **Gradients & Glows:** Used sparingly to highlight critical actions (like the 'Upload & Train' button) or to indicate real-time activity (like the pulsing gauges in the Attack Lab).
*   **Glassmorphism:** Cards and panels use semi-transparent backgrounds with subtle blurs to create depth and a modern, high-tech "Web3" feel.
*   **Responsive Flow:** The layout utilizes CSS Grid and Flexbox, ensuring the dashboard remains usable and beautiful whether viewed on a large presentation monitor or a laptop screen.
