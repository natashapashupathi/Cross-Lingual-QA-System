
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
git clone https://github.com/natashapashupathi/Cross-Lingual-QA-System/tree/main
```

## Usage

1. **Data Preparation**: Convert the QA dataset from Hindi to English and adjust the answer start positions.

2. **Training the Model**: Train the QA model with a custom dataset.

3. **Evaluation**: Evaluate the trained model to determine its performance.

4. **Prediction**: Use the provided app interface to predict answers based on given Hindi context and English questions.

### Data Preparation

The data preparation involves translating the context from Hindi to English using google translate and adjusting the answer start positions using fuzzyword.


### Training the Model

Train the model in batches using the translated dataset.

### Evaluation

Evaluate the model performance using accuracy and F1 scores.

### Prediction

Use the provided app interface for predicting answers. Here is an example of the working application

![The working app](https://github.com/natashapashupathi/Cross-Lingual-QA-System/blob/main/data/Screenshot%202024-08-04%20at%205.49.58%E2%80%AFPM.png)

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Googletrans](https://py-googletrans.readthedocs.io/en/latest/)
- [Gradio](https://www.gradio.app/)
