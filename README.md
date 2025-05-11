🎬 Sentiment Analysis on IMDB Reviews using LSTM (with Streamlit)
This project performs sentiment analysis on IMDB movie reviews using a Long Short-Term Memory (LSTM) neural network and visualizes the results with a Streamlit web app. It leverages pretrained GloVe embeddings, includes full data preprocessing, and provides a clean user interface for real-time sentiment prediction.

✅ Overview
-Movie reviews contain subjective opinions, and classifying them as positive, negative or neutral is a classic NLP task. This project:
-Preprocesses raw IMDB review text
-Uses GloVe word embeddings to represent words
-Trains an LSTM model on the dataset
-Provides a real-time prediction interface using Streamlit

🌐 Demo
You can try the app locally using:
`streamlit run app.py `


🚀 Features
1-🧠 Deep learning model using LSTM
2-🔤 Word embeddings via GloVe (100d)
3-⚙️ Clean preprocessing pipeline with stopword removal
4-📊 Visual performance metrics 
5-🖥️ Simple and responsive Streamlit UI
6-📥 Real-time review input for prediction

🛠️ Installation
1-Clone the repo:
`git clone https://github.com/ayberkpalta/LSTM-sentiment.git
cd LSTM-sentiment`
cd `lstm_app.py `


2-Install requirements:
`pip install -r requirements.txt`

Download GloVe embeddings (100d) from  `https://nlp.stanford.edu/projects/glove/  `and place it in the project directory.

▶️ Usage
To run the app:
`streamlit run app.py`

Then enter any IMDB-style movie review in the input box. The app will display whether the sentiment is positive , negative or neutral.

🧬 Model Architecture
-Embedding Layer: Initialized with 100-dimensional GloVe vectors
-LSTM Layer: 128 units
-Dropout: 0.5 to prevent overfitting
-Dense Output Layer: Sigmoid activation for binary classification
-The training data is padded to a fixed sequence length of 100 tokens.

📈 Results
-Metric	Value
-Accuracy	88.5%
-Loss	~0.28
-Optimizer	Adam
-Epochs	5





