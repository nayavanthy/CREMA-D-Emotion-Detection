import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from inference import run  # Import the run function from inference.py
import os

# Set the title of the app
st.title("Customer Care Emotion Analysis")

# File uploader to upload the audio file
uploaded_file = st.file_uploader("Upload a call recording (WAV format)", type=["wav"])

# Display a button to run inference after the file is uploaded
if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "./temp_audio.wav"
    
    # Save the file locally for processing
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run inference on the uploaded file
    predictions, score = run(temp_file_path)

    # Display the score
    st.header(f"Executive Score: {score:.2f}")

    # Time axis assuming 3-second segments (you can modify this if the segment length is different)
    time_axis = np.arange(len(predictions)) * 3  # Multiply by 3 for 3-second intervals

    # Classify emotions into positive and negative
    positive_emotions = [1, 5, 3]  # HAP, NEU
    negative_emotions = [0, 2, 4]  # SAD, ANG, FEA
    classified_emotions = ['Positive' if pred in positive_emotions else 'Negative' for pred in predictions]

    # Create a mask for the positive and negative classifications
    positive_mask = np.array([1 if em == 'Positive' else 0 for em in classified_emotions])
    negative_mask = np.array([1 if em == 'Negative' else 0 for em in classified_emotions])

    # Create a figure and axes
    plt.figure(figsize=(12, 6))

    # Fill between to create green and red tints
    plt.fill_between(time_axis, 1, where=positive_mask, color='green', alpha=0.3, label='Positive')
    plt.fill_between(time_axis, 0, where=negative_mask, color='red', alpha=0.3, label='Negative')

    # Smoothing the edges by using a rolling mean (window of 3)
    smoothed_predictions = np.convolve(positive_mask, np.ones(3)/3, mode='same')

    # Plot the smoothed values
    plt.plot(time_axis, smoothed_predictions, color='blue', linewidth=2, label='Emotion Classification')

    # Customizing the plot
    plt.title("Emotion Fluctuation Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Emotion Classification")
    plt.yticks([0, 1], ['Negative', 'Positive'])  # Labels for y-axis
    plt.xticks(time_axis)  # Optional: Customize the x-ticks
    plt.grid(True)
    plt.legend(loc='upper right')

    # Show the plot
    st.pyplot(plt)

    # Clean up the temporary file
    os.remove(temp_file_path)
