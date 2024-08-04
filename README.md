
# Crosslingual QA Project

This project aims to create a Question Answering (QA) system that can process Hindi contexts, translate them to English, and then provide answers to questions posed in English. The system utilizes a combination of translation services and machine learning models to achieve crosslingual QA capabilities.

## Features

- **Translation**: Converts Hindi context to English.
- **Position Adjustment**: Adjusts the answer start positions in the new translated context using fuzzyword.
- **Model Training**: Trains the QA model in batches.
- **Evaluation**: Evaluates the model performance.
- **Prediction**: Provides an app interface for predicting answers from Hindi context and English questions.

## Installation

To get started with the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/crosslingual-qa-project.git
cd crosslingual-qa-project
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Convert the QA dataset from Hindi to English and adjust the answer start positions.

2. **Training the Model**: Train the QA model with the prepared dataset.

3. **Evaluation**: Evaluate the trained model to determine its performance.

4. **Prediction**: Use the provided app interface to predict answers based on given Hindi context and English questions.

### Data Preparation

The data preparation involves translating the context from Hindi to English and adjusting the answer start positions using fuzzyword.

```python
from googletrans import Translator
import fuzzyword

# Initialize the Google Translator
translator = Translator()

def translate_context(context):
    translated = translator.translate(context, src='hi', dest='en')
    return translated.text

# Use fuzzyword to adjust answer start positions
# Your code for fuzzyword implementation goes here
```

### Training the Model

Train the model in batches using the translated dataset.

```python
from transformers import pipeline

# Load the QA pipeline
qa_pipeline = pipeline("question-answering")

def train_model(dataset):
    # Your code for training the model goes here
    pass
```

### Evaluation

Evaluate the model performance.

```python
def evaluate_model(model, validation_dataset):
    # Your code for evaluating the model goes here
    pass
```

### Prediction

Use the provided app interface for predicting answers.

```python
import gradio as gr
from transformers import pipeline

# Load the QA pipeline
qa_pipeline = pipeline("question-answering")

def answer_question(question, context):
    translated_context = translate_context(context)
    result = qa_pipeline(question=question, context=translated_context)
    return result["answer"]

iface = gr.Interface(
    fn=answer_question,
    inputs=["text", "text"],
    outputs="text",
    title="Crosslingual QA Model",
    description="Ask a question based on the given context",
)

iface.launch()
```

## Contributing

We welcome contributions to improve this project. Feel free to submit issues or pull requests on the [GitHub repository](https://github.com/yourusername/crosslingual-qa-project).

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Googletrans](https://py-googletrans.readthedocs.io/en/latest/)
- [Gradio](https://www.gradio.app/)
