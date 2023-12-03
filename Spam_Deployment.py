# # Import necessary libraries
# import streamlit as st
# import pandas as pd
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import StandardScaler

# # Load the model and tokenizer
# model = load_model("spam_model.h5")

# # Load the tokenizer
# with open("tokenizer.pkl", "rb") as file:
#     tokenizer = pickle.load(file)



# # Streamlit app
# def main():
#     # Sidebar with detailed information about spam and the spam filter
#     st.sidebar.title("Spam Filter Information")
#     st.sidebar.write(
#         """
#         ## Spam and Spam Filtering

#         Spam, also known as unsolicited or unwanted email, is a prevalent issue in electronic communication.
#         Spam messages often contain irrelevant or inappropriate content and can be a source of scams and phishing attacks.

#         ### About the Spam Filter

#         This spam filter utilizes a machine learning model trained to distinguish between spam (unwanted)
#         and ham (wanted) messages. The model is based on a Long Short-Term Memory (LSTM) neural network,
#         which is a type of recurrent neural network (RNN) well-suited for sequence data such as text.

#         The model has been trained on a diverse dataset containing both spam and non-spam messages to
#         generalize well to different types of content.

#         ### How to Use

#         1. Enter a message in the text box.
#         2. Click the "Predict" button.
#         3. The model will predict whether the message is likely spam or ham.

#         ---
#         **Note:** This application is for demonstration purposes, and the model's predictions may not be perfect.
#         """
#     )

#     st.title("Spam Filter App")

#     # Input text box for user input
#     user_input = st.text_area("Enter a message:")

#     if st.button("Predict"):
#         # Preprocess the user input
#         user_input = [user_input]
#         sequences = tokenizer.texts_to_sequences(user_input)
#         data = pad_sequences(sequences, maxlen=20)

    
#         # Make predictions
#         prediction = model.predict(data)

#         # Display the result
#         result = "Spam" if prediction[0, 1] > 0.5 else "Ham"
#         st.success(f"The message is likely {result}.")

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model("spam_model.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Streamlit app
def main():
    # Sidebar with detailed information about spam and the spam filter
    st.sidebar.title("Spam Filter Information")
    st.sidebar.write(
        """
        ## Spam and Spam Filtering

        Spam, also known as unsolicited or unwanted email, is a prevalent issue in electronic communication.
        Spam messages often contain irrelevant or inappropriate content and can be a source of scams and phishing attacks.

        ### About the Spam Filter

        This spam filter utilizes a machine learning model trained to distinguish between spam (unwanted)
        and ham (wanted) messages. The model is based on a Long Short-Term Memory (LSTM) neural network,
        which is a type of recurrent neural network (RNN) well-suited for sequence data such as text.

        The model has been trained on a diverse dataset containing both spam and non-spam messages to
        generalize well to different types of content.

        ### How to Use

        1. Enter a message in the text box.
        2. Click the "Predict" button.
        3. The model will predict whether the message is likely spam or ham.

        ---
        **Note:** This application is for demonstration purposes, and the model's predictions may not be perfect.
        """
    )

    st.title("Spam Filter App")

    # Input text box for user input
    user_input = st.text_area("Enter a message:")

    if st.button("Predict"):
        if user_input:
            # Preprocess the user input
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=20)

            # Make predictions
            prediction = model.predict(padded_sequence)

            # Display the result
            result = "Spam" if prediction[0, 1] > 0.5 else "Ham"
            st.success(f"The message is likely {result}.")
            
    #Clear button
    if st.button("Clear"):
        user_input = ""




# Run the Streamlit app
if __name__ == "__main__":
    main()

