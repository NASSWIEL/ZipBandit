# Contextual Bandit Agent for ZipVoice Prompt Selection

## Overview
This repository contains the implementation of a Reinforcement Learning (RL) agent designed to optimize prompt selection for the ZipVoice Text-to-Speech model. The agent utilizes a Contextual Bandit approach to learn the mapping between input text and the optimal audio prompt embedding, aiming to minimize the Character Error Rate (CER) of the generated audio. The system operates in a one-shot learning manner, updating its policy based on the quality of the generated speech.

## Structure
The project is organized as follows:

```
Agent/
├── assess_CER/             # Scripts for calculating CER and Reward
│   ├── calculate_cer.py    # Computes CER using Whisper ASR
│   └── weighted_cer.py     # Converts CER to a reward signal
├── DB/                     # Vector Database generation scripts
│   ├── generate_embeddings.py      # Generates FAISS index from audio
│   ├── generate_raw_embeddings.py  # Generates raw embeddings
│   ├── run_generate_embeddings.sh  # Shell script to run embedding generation
│   └── run_generate_raw_embeddings.sh
├── generate_audio/         # Interface for ZipVoice inference
│   └── generate_with_zipVoice.py
├── logs_agent/             # Directory for SLURM execution logs
├── model/                  # Neural Network definitions and training logic
│   ├── __init__.py
│   ├── agent_model.py      # MLP Agent architecture (1024 -> 100)
│   ├── text_encoder.py     # SONAR Text Encoder wrapper
│   ├── train_agent.py      # Training step implementation
│   └── utils.py            # Utility functions
├── Similarity/             # Vector similarity search
│   └── asess_similarty.py  # FAISS search logic
├── temp_pipeline/          # Temporary storage for pipeline artifacts
├── requirements.txt        # Python dependencies
└── run_pipeline.sh         # Main SLURM pipeline script
```

## Requirements

To set up the environment, follow these steps:

```bash
# Create and activate the environment
conda create -n agent_env python=3.10
conda activate agent_env

# Install dependencies
pip install -r requirements.txt
```

**Note:** The pipeline assumes the existence of a separate environment `zipvoice_py311` for the ZipVoice inference model, as configured in `generate_with_zipVoice.py`.

## Run Pipeline

To run the full RL pipeline on a specific French sentence using SLURM:

```bash
sbatch run_pipeline.sh "Votre phrase en français ici"
```

This script will execute the 7-step pipeline for a defined number of iterations (default: 10), performing inference, evaluation, and model updates in a loop.

## Prerequisites

### Data Source
The agent is trained on the **NEB** speaker subset of the **Blizzard Challenge 2023** dataset. This dataset consists of approximately 64,000 audio segments ranging from 3 to 7 seconds.

### Vector Database
We have created a vector database of audio prompts to serve as the action space for the bandit. The database generation process is handled by `Agent/DB/generate_embeddings.py`:

1.  **Encoding**: SONAR Speech Encoder (`sonar_speech_encoder_fra`) encodes each audio prompt into a 1024-dimensional vector.
2.  **Dimensionality Reduction**: PCA reduces these vectors from 1024 to 100 dimensions to create a compact latent space for the agent to learn.
3.  **Storage**:
    *   `pca_model.pkl`: A serialized `sklearn.decomposition.PCA` object.
        *   **Input Dimension**: 1024 (SONAR embedding size)
        *   **Output Dimension**: 100 (Latent space size)
        *   **Components**: Stores the projection matrix `(100, 1024)` and mean vector `(1024,)`.
    *   `prompts.index`: The FAISS index (`faiss.IndexFlatL2`) used for fast similarity search.
        *   **Total Vectors**: ~63,478
        *   **Dimension**: 100
        *   **Metric**: L2 Distance (equivalent to Cosine Similarity on normalized vectors).
    *   `prompts_metadata.pkl`: A serialized Python list containing metadata for each vector in the index (aligned by index position).
        *   **Structure**: List of dictionaries.
        *   **Item Example**:
            ```python
            {
                'wav_name': 'ES_LMP_NEB_01_0001_24592_25697',
                'prompt_transcription': 'Le tapis-franc...',
                'prompt_wav': '/info/corpus/Blizzard2023_segmented/segmented/NEB_train/...'
            }
            ```

## Details in Functioning

The pipeline consists of 7 steps that repeat for every iteration to simulate online learning:

**Step 1: Text Encoding**
The input French sentence is encoded using the **SONAR Text Encoder** (`text_sonar_basic_encoder`). This produces a 1024-dimensional semantic vector representation of the target text.

**Step 2: Agent Prediction**
The **Agent Model** (a Multi-Layer Perceptron) takes the 1024-dim text vector as input and predicts a 100-dimensional vector. This output represents the "ideal" prompt embedding in the reduced latent space that the agent believes will yield the best speech synthesis for the given text.

**Step 3: Similarity Search**
The system performs a Cosine Similarity search using **FAISS** between the agent's predicted vector and the pre-computed database of prompt embeddings. It retrieves the nearest neighbor (the most similar existing audio prompt) and returns its ID, WAV path, and transcription.

**Step 4: Audio Generation**
Using the retrieved prompt (audio and text) and the original target text, the **ZipVoice** model generates the corresponding speech audio.

**Step 5: CER Calculation**
The generated audio is transcribed using **OpenAI Whisper (Large V3)**. The transcription is compared against the original target text to calculate the Character Error Rate (CER).

**Step 6: Reward Calculation**
The CER is converted into a reward signal (Scalar [0, 1]). A lower CER results in a higher reward (e.g., `Reward = max(0, 1 - CER)`).

**Step 7: Agent Update (Contextual Bandit)**
The agent is updated using the collected experience tuple `(State, Action, Reward)`.

*   **State**: Input text vector (1024-dim).
*   **Action**: The actual vector of the retrieved prompt (100-dim).
*   **Reward**: The calculated reward.

The model minimizes the weighted distance between its prediction and the successful prompt vector, effectively reinforcing the selection of prompts that lead to high-quality audio generation.

### Agent Model Structure

The agent is implemented as a **Multi-Layer Perceptron (MLP)** using PyTorch, designed to map the semantic space of input text to the latent space of audio prompts.

*   **Input Layer**: Accepts a **1024-dimensional** vector (SONAR Text Embedding).
*   **Hidden Layers**:
    1.  **Linear** (1024 $\to$ 512) $\to$ **BatchNorm1d** $\to$ **ReLU** $\to$ **Dropout** (0.2)
    2.  **Linear** (512 $\to$ 256) $\to$ **BatchNorm1d** $\to$ **ReLU** $\to$ **Dropout** (0.2)
*   **Output Layer**: **Linear** (256 $\to$ 100). Outputs a raw continuous vector representing the coordinates in the prompt latent space.

### The Reinforcement Learning Policy

The system utilizes a **Contextual Bandit** formulation to optimize the prompt selection policy in an online setting.

*   **State ($s$)**: The 1024-dim embedding of the input text.
*   **Action ($a$)**: The 100-dim embedding of the audio prompt.
*   **Policy ($\pi_\theta$)**: A deterministic neural network that predicts an ideal action vector $\hat{a} = \pi_\theta(s)$. The actual action taken is the **Nearest Neighbor** of $\hat{a}$ in the pre-computed vector database.
*   **Update Rule (Reward-Weighted Regression)**:
    The model parameters $\theta$ are updated to minimize the weighted Mean Squared Error (MSE) between the predicted vector $\hat{a}$ and the executed action vector $a_{retrieved}$, scaled by the observed reward $R$.
    
$$
\mathcal{L}(\theta)
= R \cdot \lVert \pi_\theta(s) - a_{\text{retrieved}} \rVert^{2}
$$

    *   **Interpretation**: If a prompt yields a high reward (low CER), the agent updates its weights to predict vectors closer to that prompt in the future. If the reward is low, the gradient is scaled down, effectively ignoring poor examples.

### File Descriptions

*   **`run_pipeline.sh`**: Main script running the 7-step RL loop on SLURM.
*   **`model/agent_model.py`**: MLP Agent (1024 $\to$ 100 dim) for prompt prediction.
*   **`model/text_encoder.py`**: Generates 1024-dim text embeddings using SONAR.
*   **`model/train_agent.py`**: Performs the Contextual Bandit gradient update.
*   **`DB/generate_embeddings.py`**: Creates the FAISS vector index from audio prompts.
*   **`Similarity/asess_similarty.py`**: Finds the nearest prompt via Cosine Similarity.
*   **`generate_audio/generate_with_zipVoice.py`**: Runs ZipVoice inference for TTS.
*   **`assess_CER/calculate_cer.py`**: Calculates CER using OpenAI Whisper.
*   **`assess_CER/weighted_cer.py`**: Computes the reward signal from CER.
