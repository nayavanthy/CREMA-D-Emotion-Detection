# Emotion Detection in Speech

This project is a deep learning-based system designed to analyze customer service call recordings. It evaluates the emotional state of customers and assigns a performance score to customer service executives based on their ability to de-escalate emotional distress.

## Objective
Provide a metric for assessing customer service effectiveness through emotion detection.

## Dataset
- **CREMA-D Dataset**: Audio recordings labeled with six emotions—anger, disgust, fear, happy, neutral, and sadness.

## Key Components
### 1. **Data Preprocessing**
   - **Feature Extraction**: Mel-Frequency Cepstral Coefficients (MFCCs) and mel-spectrograms.
   - **Data Augmentation**: Techniques like time-stretching ensure uniform 3-second clips.

### 2. **Models**
   - **CNN**: Baseline model achieving 43.7% test accuracy.
   - **Transformer**: Achieved 50.7% test accuracy but needs further optimization.
   - **CNN+LSTM (Hybrid)**: Captures spatial and temporal patterns with moderate performance.
   - **Wav2Vec 2.0**: A transformer-based pre-trained model for raw audio.

### 3. **Deployment**
   - **Web Application**: Built using Streamlit for interactive analysis.
   - **Scripts**:
      - `train_data_preprocessing.py`: Prepares training data.
      - `inference_data_preprocessing.py`: Prepares audio for inference.
      - `inference.py`: Runs predictions and scoring.
      - `model.py`: Defines and trains models.
      - `app.py`: Hosts the Streamlit app.

## Notebooks
- **modelling.ipynb**: Contains model definitions, training details, and visualizations.

## Deployment
1. Install dependencies: `pip install -r requirements.txt`
2. Preprocess data: `python train_data_preprocessing.py`
3. Train models: `python model.py`
4. Launch web app: `streamlit run app.py`

## Results
- **Best Accuracy**: Transformer model with 50.7% test accuracy.
- **Visualizations**: Real-time emotion tracking during calls with positive/negative classifications.

## Future Work
- Enhance model generalization and accuracy.
- Integrate advanced data balancing and regularization techniques.

## License
This project is licensed under [Your License].

## Author
- **Nayavanth Yogi**
