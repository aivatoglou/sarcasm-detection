# Sarcasm detector

A simple sarcasm detector trained on [News-Headlines-Dataset-For-Sarcasm-Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection). The dataset was collected from TheOnion and HuffPost. The architecture is based on [DistilBERT](https://arxiv.org/abs/1910.01108), which was fine-tuned for binary classification between sarcastic or not headlines. Moreover, the trained model was converted to [ONNX format](https://github.com/onnx/onnx) and was quantized using the [ONNX runtime](https://onnxruntime.ai/docs/performance/quantization.html). Finally, [streamlit](https://streamlit.io/) was used for creating a simple web app that asks for user input and depicts the classification results and inference time.

## Usage
To install the dependencies:
```bash
pip install -r requirements.txt
```
### - Train
To create the dataset:
```bash
cd src/data
```
```python
python make_dataset.py 
```
To start fine-tuning:
```bash
cd src
```
```python
python main.py
```

### - Inference 
```bash
streamlit run src/models/inference.py
```
