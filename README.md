# Edge AI Prototype

## Task 1: Recyclable Item Classification
- `download_dataset.py`: Downloads Kaggle dataset (`techsash/waste-classification-data`).
- `train_model.py`: Trains MobileNetV2 on ORGANIC/RECYCLABLE classes.
- `convert_model.py`: Converts to TFLite.
- `test_model.py`: Tests TFLite model.
- `training_history.txt`: Accuracy metrics.
- `accuracy_plot.png`: Accuracy plot.
- `recyclable_model.h5`: [Google Drive](https://drive.google.com/file/d/abc123/view?usp=sharing)
- `recyclable_classifier.tflite`: [Google Drive](https://drive.google.com/file/d/xyz789/view?usp=sharing)

## Task 2: Smart Agriculture System
- `smart-agriculture/download_crop_dataset.py`: Downloads `patelris/crop-yield-prediction-dataset`.
- `smart-agriculture/predict_yield.py`: Predicts crop yields.
- `smart-agriculture/yield_metrics.txt`: Model metrics.
- `smart-agriculture/data_flow_diagram.png`: System diagram.
- `smart-agriculture/proposal.md`: Proposal document.
- `smart-agriculture/proposal.pdf`: Proposal PDF.

## Task 3: Ethics in Personalized Medicine
- `ethics_analysis.md`: Ethical analysis.
- `ethics_analysis.pdf`: Ethical analysis PDF.

## Setup
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python download_dataset.py`
5. `python train_model.py`
6. Download models from Google Drive links
7. `python convert_model.py`
8. `python test_model.py`
9. `python smart-agriculture/download_crop_dataset.py`
10. `python smart-agriculture/predict_yield.py`

## Project Structure

~/Desktop/plp/edge-ai-prototype/
├── smart-agriculture/
│   ├── download_crop_dataset.py
│   ├── predict_yield.py
│   ├── yield_metrics.txt
│   ├── data_flow_diagram.png
│   ├── proposal.md
│   ├── proposal.pdf
├── download_dataset.py
├── train_model.py
├── convert_model.py
├── test_model.py
├── test_image.jpg
├── training_history.txt
├── accuracy_plot.png
├── task1_report.md
├── task1_report.pdf
├── ethics_analysis.md
├── ethics_analysis.pdf
├── AI-Powered Climate-Adaptive Urban Planning for 2030 (Part 3: Futuristic Proposal)
├── Pioneering Tomorrow's AI Innovations.pdf (Part 1: Theoretical Analysis)
├── part2_report.pdf
├── requirements.txt
├── .gitignore
├── README.md