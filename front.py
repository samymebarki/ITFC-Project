
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from streamlit_image_comparison import image_comparison

# Set page config
st.set_page_config(page_title="TextSage - AI Model Prediction", page_icon="💻", layout="wide")

# Center Content with Custom CSS
st.markdown("""
    <style>
        /* Center all content */
        .block-container {
            max-width: 1000px; /* Adjust width for readability */
            margin: 0 auto; /* Center align */
        }
        /* Center sidebar logo */
        [data-testid="stSidebar"] .css-1d391kg {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        /* Adjust Lottie animation alignment */
        iframe {
            display: block;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)





# Sidebar with Logo
st.sidebar.image("logo.png", width=300)
st.sidebar.title("Navigation")
sidebar_option = st.sidebar.selectbox("Select a section", ["Home", "Text Prediction", "Batch Prediction"])

# Home Page
if sidebar_option == "Home":
    # Add a centered title
    st.markdown('<h1 style="text-align:center; color:#0fc1d1;">Welcome to TextSage!👋</h1>', unsafe_allow_html=True)
    
    # Add introductory text with center alignment
    st.markdown("""
        <div style="color:#e6e6e6; font-family: monospace; text-align:center;">
            <p>TextSage is an advanced AI-powered tool designed to predict whether a given text is written by a human or AI.</p>
            
        </div>
    """, unsafe_allow_html=True)
    image_comparison(
    img1="single.jpg",
    img2="batch.jpg",
    label1="S",
    label2="B",
)
    # Display the image from the current directory
    #st.image("diagram.jpg", use_container_width=True)

# Load Model and Tokenizer
try:
    loaded_model = load_model('my_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# Text Prediction Section
if sidebar_option == "Text Prediction":
    st.markdown('<h2 style="text-align:center;">Text Prediction</h2>', unsafe_allow_html=True)
    input_text = st.text_area("Enter text for prediction:")

    if st.button("Predict"):
        if not input_text.strip():
            st.warning("Please enter some text for prediction.")
        else:
            sequences = tokenizer.texts_to_sequences([input_text])
            max_vocab_index = loaded_model.get_layer(index=0).input_dim - 1
            sequences = [[min(idx, max_vocab_index) for idx in seq] for seq in sequences]
            max_sequence_length = loaded_model.input_shape[1]
            padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
            prediction = loaded_model.predict(padded_sequence)[0][0]

            st.subheader("Prediction Result")
            if prediction > 0.90:
                st.write(f"Human-authored with very high probability: {prediction * 100:.2f}%")
            elif prediction > 0.69:
                st.write(f"Human-authored with high probability: {prediction * 100:.2f}%")
            elif prediction > 0.51:
                st.write(f"AI-assisted with low probability of human authorship: {prediction * 100:.2f}%")
            else:
                st.write(f"AI-authored with very low probability of human authorship: {prediction * 100:.2f}%")

# Batch Prediction Section
# Batch Prediction Section
if sidebar_option == "Batch Prediction":
    # Section Title
    st.markdown('<h2 style="text-align:center;">Batch Prediction</h2>', unsafe_allow_html=True)

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file for batch predictions", type=["csv"])

    if uploaded_file:
        try:
            # Load the data
            test_data = pd.read_csv(uploaded_file)

            # User input for column names
            st.write("Please specify the column names for the text to be predicted and the actual labels (optional for accuracy):")
            text_column = st.text_input("Column name for text:", value="text")
            label_column = st.text_input("Column name for actual labels (optional for accuracy):", value="label")

            if text_column not in test_data.columns:
                st.error(f"The uploaded CSV must have a column named '{text_column}'.")
            else:
                # Preprocess the text column
                max_sequence_length = loaded_model.input_shape[1]
                max_vocab_index = loaded_model.get_layer(index=0).input_dim - 1
                sequences = tokenizer.texts_to_sequences(test_data[text_column])
                sequences = [[min(idx, max_vocab_index) for idx in seq] for seq in sequences]
                padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

                # Make predictions
                batch_predictions_prob = loaded_model.predict(padded_sequences)
                test_data['predictions'] = (batch_predictions_prob > 0.5).astype(int)  # Binary predictions

                # Display batch predictions
                st.subheader("Batch Prediction Results")
                st.write(test_data)

                # Download the predictions
                csv = test_data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

                # Optional: Calculate and display accuracy if 'generated_column' is provided
                if label_column in test_data.columns:
                    from sklearn.metrics import accuracy_score

                    # Ensure actual labels are binary
                    actual_labels = test_data[label_column].astype(int)
                    predictions = test_data['predictions']

                    # Calculate accuracy
                    accuracy = accuracy_score(actual_labels, predictions)

                    # Display accuracy
                    st.subheader("Model Accuracy")
                    st.write(f"Accuracy of the model on the dataset: **{accuracy * 100:.2f}%**")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("""
    <style>
        .footer {
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    fontweight: bold;
                    width: 120%;
                    color: rgba(0, 0, 0, 0.9); /* Light blue with 60% opacity */
                    text-align: center;
                    font-family: monospace;
                }
    </style>
    <div class="footer">
        <p> Made by Hani and Samy 👻 --> ITFC Project </p>
    </div>
""", unsafe_allow_html=True)