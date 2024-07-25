import numpy as np
import heartpy as hp

# This is the code to analyze user emotion through heart rate.
# The HeartPy library extracts human heartbeat and takes 10 beats in 1 beat per second format.
# It then calculates SDNN, RMSSD, and LF/HF ratio.
# By analyzing these patterns, human emotion can be assessed.

# Here's a synthetic heart rate signal with varying heart rates over time
heart_rate_data = [70, 72, 75, 80, 85, 88, 90, 92, 95, 100]

try:
    # Process the heart rate signal to extract features
    sample_rate = 10  # Reduced sample rate
    wd, m = hp.process(hp.scale_data(heart_rate_data), sample_rate)

    # Extract relevant features
    sdnn = m.get('sdnn', None)  # Standard deviation of NN intervals
    rmssd = m.get('rmssd', None)  # Root mean square of successive differences
    lf_hf_ratio = m.get('lf/hf', None)  # Low-frequency to high-frequency ratio (LF/HF ratio)

    # Define emotional states and conditions
    calm_condition = sdnn is not None and rmssd is not None and lf_hf_ratio is not None and sdnn > 50 and rmssd > 20 and lf_hf_ratio < 0.5
    happy_condition = sdnn is not None and lf_hf_ratio is not None and sdnn > 50 and lf_hf_ratio > 0.5
    sad_condition = rmssd is not None and lf_hf_ratio is not None and rmssd < 20 and lf_hf_ratio < 0.5
    angry_condition = sdnn is not None and lf_hf_ratio is not None and sdnn < 50 and lf_hf_ratio > 1.0
    surprised_condition = sdnn is not None and rmssd is not None and lf_hf_ratio is not None and sdnn < 50 and rmssd < 20 and lf_hf_ratio > 0.8

    # Determine emotional state based on conditions
    if calm_condition:
        emotion = "Calm"
    elif happy_condition:
        emotion = "Happy"
    elif sad_condition:
        emotion = "Sad"
    elif angry_condition:
        emotion = "Angry"
    elif surprised_condition:
        emotion = "Surprised"
    else:
        emotion = "Neutral"

    print("Estimated Emotion:", emotion)

except Exception as e:
    print("Error:", e)
    print("Could not determine emotional state.")
