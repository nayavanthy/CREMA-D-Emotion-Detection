import inference_data_processing
from model import TransformerBlock, PositionalEmbedding

import tensorflow as tf

SAMPLE_CALL = "./sample_call.wav"

def run_model(segments):
    model = tf.keras.models.load_model('emotion_recognition_model.keras', custom_objects={'TransformerBlock': TransformerBlock})

    emotion_predictions = []

    for segment in segments:
        emotion_prediction = model.predict(segment.reshape((1, segment.shape[0], segment.shape[1])))

        predicted_class_index = emotion_prediction.argmax()

        emotion_predictions.append(predicted_class_index)

    return emotion_predictions

def score_executive(mapped_emotions):
    # Base score starts at 5
    score = 5
    
    positive_emotions = [1, 3, 5]  # HAP, NEU, DIS
    negative_emotions = [0, 2, 4]  # SAD, ANG, FEA
    
    # Iterate through the mapped emotions and adjust the score
    for i in range(1, len(mapped_emotions)):
        prev_emotion = mapped_emotions[i - 1]
        curr_emotion = mapped_emotions[i]
        
        if curr_emotion in negative_emotions:
            score -= 0.1
        else:
            score += 0.1
        # Improve score if moving to a more positive emotion
        if prev_emotion in negative_emotions and curr_emotion in positive_emotions:
            score += 0.5  # Positive transition
        # Decrease score if moving to a more negative emotion
        elif prev_emotion in positive_emotions and curr_emotion in negative_emotions:
            score -= 0.5  # Negative transition

    # Ensure score is within the range 1 to 10
    score = max(1, min(10, score))
    
    return score

def run(input_file):
    
    mfcc_segments = inference_data_processing.data_processing(input_file)

    predictions = run_model(mfcc_segments)

    score = score_executive(predictions)

    return predictions, score

if __name__ == "__main__":
    _ , score = run(SAMPLE_CALL)
    print(score)