# Reverse Dictionary

This project implements a "Reverse Dictionary" application using Streamlit, allowing users to find words based on their descriptions. It leverages two different approaches for word retrieval: a vector search-based method and an LLM (Large Language Model) based method.

## Features

- **Streamlit Interface**: An interactive web application for easy input and display of results.
- **Vector Search**: Utilizes Pinecone as a vector database and Google's `embedding-001` model to find words semantically similar to the user's description.
- **LLM-Based Word Generation**: Employs the `gemini-2.0-flash` model to generate words and their definitions directly from a textual description.
- **Side-by-Side Comparison**: Displays results from both vector search and LLM methods for comprehensive understanding.

## Setup

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.8+
- uv (recommended for dependency management) or pip
- Access to Google Generative AI API (for `gemini-2.0-flash` and `embedding-001`)
- Access to Pinecone (for vector database)

### Environment Variables

Create a `.env` file in the root directory of the project and add your API keys:

```/dev/null/example.env#L1-2
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/EngineerLambda/reverse-dict-v2.git
    cd reverse-dict-v2
    ```

2.  **Install dependencies**:
    Using `uv` (as indicated by `uv.lock`):
    ```bash
    uv sync
    ```
    Alternatively, using `pip` with `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation (Vector Database Population)

The vector search functionality requires a pre-populated Pinecone index. A `resources/data.pkl` file containing word descriptions is expected. If you have this file, you can run the `vectordb.py` script to populate your Pinecone index.

1.  Ensure you have your `resources/data.pkl` file in the `resources/` directory.
2.  Run the `main` function in `vectordb.py` to add the documents to your Pinecone store:

    ```bash
    python reverse-dict-v2/vectordb.py
    ```

    This script will initialize the Pinecone index (if it doesn't exist) and then add the word-definition pairs from `data.pkl` to it.

## Usage

Once the setup is complete and the vector database is populated, you can run the Streamlit application:

```bash
streamlit run reverse-dict-v2/app.py
```

This will open the application in your web browser, usually at `http://localhost:8501`.

Enter a description of the word you're thinking of into the text input field and click "Find word". The application will display results from both the Vector Search and LLM-Based methods.
