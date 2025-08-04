import streamlit as st
import nltk

# --- Always Download NLTK Data at the Very Top ---
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Pre-computation and Data Loading ---
@st.cache_data
def load_and_preprocess_data(file_path='grimm_tales.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().replace('the juniper-tree.', 'the juniper-tree')
    sent_tokens = nltk.sent_tokenize(data)
    return sent_tokens

# --- Core Chatbot Functions ---
lemmatizer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower()))

def chatbot_response(user_input, sent_tokens):
    bot_response = ''
    temp_tokens = list(sent_tokens)
    temp_tokens.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(temp_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    similar_indices = vals.argsort()[0][-6:-1]
    best_idx = -1
    for idx in reversed(similar_indices):
        sentence = temp_tokens[idx]
        if not (sentence.isupper() and len(sentence.split()) < 5):
            best_idx = idx
            break
    if best_idx == -1:
        return "I am sorry! I don't understand you. Please ask a more specific question about the stories."
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        bot_response = "I am sorry! I don't understand you. Please ask a question about Grimm's Fairy Tales."
    else:
        bot_response = temp_tokens[best_idx]
        if (best_idx + 1) < len(temp_tokens) -1:
            bot_response += " " + temp_tokens[best_idx + 1]
    return bot_response

# --- Streamlit Application Main Function ---
def main():
    st.title("Chatbot: Grimm's Fairy Tales ")
    st.write("Ask me questions about Grimm's Fairy Tales by typing below.")

    sent_tokens = load_and_preprocess_data()

    if 'response' not in st.session_state:
        st.session_state.response = ""
    if 'input' not in st.session_state:
        st.session_state.input = ""

    user_text_input = st.text_input("Type your message here:", key="text_input")

    if user_text_input and user_text_input != st.session_state.input:
        st.session_state.input = user_text_input
        st.session_state.response = chatbot_response(st.session_state.input, sent_tokens)

    if st.session_state.response:
        st.text_area("Chatbot:", value=st.session_state.response, height=200)

if __name__ == "__main__":
    main()