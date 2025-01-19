import train_data_processing

import numpy as np 
import pickle

def data_processing(input_file):

    audio_data, sample_rate = train_data_processing.load_audio(input_file)

     # Calculate number of samples per segment
    segment_samples = 3 * sample_rate
    
    # Get total number of samples in the audio
    total_samples = len(audio_data)
    
    # Prepare a list to hold the MFCC features for each segment
    mfcc_segments = []

    with open('scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)

    for i in range(0, total_samples, segment_samples):
        # Get the start and end of the segment
        start_sample = i
        end_sample = min(i + segment_samples, total_samples)
        
        # Extract the segment
        segment = audio_data[start_sample:end_sample]
        
        # Pad the last segment if it's shorter than 3 seconds
        if len(segment) < segment_samples:
            padding = np.zeros(segment_samples - len(segment))
            segment = np.concatenate((segment, padding))

        mfcc = train_data_processing.extract_features(segment, sample_rate)

        mfcc = np.concatenate(list(mfcc.values()))

        scaled_mfcc = loaded_scaler.transform(mfcc.reshape(1, -1))

        mfcc_segments.append(scaled_mfcc)

    return mfcc_segments

if __name__ == "__main__":
    _ = data_processing("./sample_call.wav")
    print("done")



    

