import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import pickle

AUDIO_DIR = "./data/AudioWAV/"

def load_audio(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)  # Load with original sample rate
    return audio_data, sample_rate

def extract_features(audio_data, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    
    features = {
        'mfcc': mfcc.mean(axis=1),
        'chroma': chroma.mean(axis=1),
        'spectral_contrast': spectral_contrast.mean(axis=1),
    }
    return features

def pad_or_trim(audio_data, target_length):
    if len(audio_data) > target_length:
        return audio_data[:target_length]
    else:
        return librosa.util.fix_length(audio_data, size= target_length)

def extract_label(file_name):
    # Example: '1091_TSI_SAD_XX.wav'
    return file_name.split('_')[2]  # 'SAD'

emotion_mapping = {
    'SAD': 0,
    'HAP': 1,
    'ANG': 2,
    'DIS': 3,
    'FEA': 4,
    'NEU': 5
}

def label_to_int(label):
    return emotion_mapping.get(label, -1)

def scale_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return scaled_features

def preprocess_dataset(dataset_dir, target_length):
    all_features = []
    all_labels = []
    
    # Get list of all audio files in the dataset directory
    audio_files = [f for f in os.listdir(dataset_dir) if f.endswith('.wav')]

    # Loop over all audio files in the dataset directory with progress bar
    for file_name in tqdm(audio_files, desc="Processing Audio Files"):
        file_path = os.path.join(dataset_dir, file_name)
        
        # Step 1: Load the audio
        audio_data, sample_rate = load_audio(file_path)

        # Step 2: Pad/trim to desired length
        audio_data = pad_or_trim(audio_data, target_length)

        # Step 3: Extract features
        features = extract_features(audio_data, sample_rate)

        # Step 4: Get label from filename
        emotion_label = extract_label(file_name)
        label = label_to_int(emotion_label)

        # Step 5: Collect features and labels
        all_features.append(np.concatenate(list(features.values())))  # Flatten features into one array
        all_labels.append(label)

    return np.array(all_features), np.array(all_labels)

def save_features(features, labels, output_path):
    np.save(output_path + '_features.npy', features)
    np.save(output_path + '_labels.npy', labels)

if __name__ == "__main__":
    # Now call the preprocessing function
    features, labels = preprocess_dataset(AUDIO_DIR, 22050 * 3)

    features_scaled = scale_features(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    save_features(X_train, y_train, 'train_data')
    save_features(X_test, y_test, 'test_data')