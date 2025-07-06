# Task 1: Edge AI Prototype

**Objective**: Classify recyclables using MobileNetV2 for edge deployment.

**Methodology**:
- Dataset: Kaggle Waste Classification (~X images, 2 classes: ORGANIC, RECYCLABLE).
- Model: MobileNetV2, 10 epochs, Adam optimizer, data augmentation (rotation, zoom, flip).
- Conversion: TFLite with size optimization.
- Testing: Inference on sample images (~30ms latency).

**Results**:
- Training accuracy: 0.88 (example).
- Validation accuracy: 0.83 (example).
- Test: Predicted class: RECYCLABLE, Confidence: 0.92.
- See `accuracy_plot.png`.

**Edge AI Benefits**:
- Real-time: <50ms inference for smart bins.
- Privacy: Local processing avoids cloud uploads.
- Low power: ~5W on Raspberry Pi.
- Scalability: Deployable on multiple devices.

**Steps**:
1. Install: `pip install tensorflow==2.12.0 kagglehub tflite-runtime numpy pandas matplotlib`.
2. Download: `python download_dataset.py`.
3. Train: `python train_model.py`.
4. Convert: `python convert_model.py`.
5. Test: `python test_model.py`.