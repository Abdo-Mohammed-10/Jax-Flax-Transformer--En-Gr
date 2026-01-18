# Transformer EN - > GR with JAX & Flax 🚀
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/اسم-حسابك/اسم-الريبو/blob/main/transformer_jax.ipynb](https://colab.research.google.com/drive/1Y3HIB7abTZqS4ggOcyEEIxS-Lk09dZ54?usp=sharing))
[![JAX](https://img.shields.io/badge/JAX-Accelerated%20Linear%20Algebra-blue)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-Neural%20Network%20Library-green)](https://github.com/google/flax)
[![Optax](https://img.shields.io/badge/Optax-Optimization-red)](https://github.com/deepmind/optax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete implementation of the **Transformer** architecture for Neural Machine Translation (NMT), built using the **JAX** ecosystem (**Flax** for neural networks and **Optax** for optimization).

This project demonstrates how to build, train, and evaluate a sequence-to-sequence model to translate between **English and German** using the **Multi30k** dataset.

## 🌟 Key Features

* **High Performance:** Utilizes JAX's XLA compilation for accelerated training on GPUs/TPUs.
* **Modern Stack:** Built with Flax Linen API and Optax optimizers.
* **Preprocessing:** Uses `spaCy` for tokenization and `torchtext`/`datasets` for data handling.
* **Monitoring:** Includes training loss visualization and BLEU score evaluation.

## 🛠️ Technologies Used

* **Language:** Python 3.x
* **Deep Learning:** [JAX](https://github.com/google/jax), [Flax](https://github.com/google/flax)
* **Optimization:** [Optax](https://github.com/deepmind/optax)
* **NLP & Data:** [Hugging Face Datasets](https://huggingface.co/docs/datasets), [spaCy](https://spacy.io/), [NLTK](https://www.nltk.org/)

## 📂 Dataset

The model is trained on the **[Multi30k](https://github.com/multi30k/dataset)** dataset, a standard benchmark for machine translation tasks (German-English).

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment or Google Colab (with GPU runtime).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/transformer-jax-flax.git](https://github.com/your-username/transformer-jax-flax.git)
    cd transformer-jax-flax
    ```

2.  **Install dependencies:**
    ```bash
    pip install jax jaxlib flax optax datasets spacy torch tqdm matplotlib nltk
    ```

3.  **Download spaCy language models:**
    ```bash
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
    ```

## 🏃‍♂️ Usage

You can run the project directly using the provided Jupyter Notebook.

1.  Open `Transformer___EN_GR___JAX_FLAX_&_OPTAX.ipynb` in Jupyter or Google Colab.
2.  Run the cells sequentially to:
    * Load and preprocess the Multi30k dataset.
    * Initialize the Transformer model architecture.
    * Train the model (Optimization loops included).
    * Evaluate performance using Loss and BLEU metrics.

## 📊 Training Results

The model employs `CrossEntropyLoss` and tracks performance over epochs.
* **Optimizer:** AdamW (via Optax) with learning rate scheduling.
* **Loss:** significantly decreases significantly over the first 10 epochs (e.g., from ~4.9 to ~2.0).

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
