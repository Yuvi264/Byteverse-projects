from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, emit
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import random
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Socket.io
socketio = SocketIO(app, cors_allowed_origins="*")

# ====================== MENTOR MATCHING SYSTEM ======================

# Load mentor data
# with open('mentors.json', 'r') as f:
#     mentors = json.load(f)
try:
    with open('mentors.json', 'r') as f:
        mentors = json.load(f)
    
    # Ensure each mentor has problem_tags (default to empty list if missing)
    for mentor in mentors:
        mentor['problem_tags'] = mentor.get('problem_tags', [])
# Prepare ML model for mentor recommendations
    vectorizer = TfidfVectorizer()
    mentor_tags = [' '.join(m['problem_tags']) for m in mentors]
    tfidf_matrix = vectorizer.fit_transform(mentor_tags)

except Exception as e:
    print(f"Error loading or processing mentor data: {e}")
    mentors = [
        {
            "id": "mentor1",
            "name": "Default Mentor",
            "problem_tags": ["anxiety", "stress"],
            "rating": 4.5,
            "story": "Experienced in helping with anxiety and stress management",
            "language": "English"
        }
    ]
    mentor_tags = [' '.join(m['problem_tags']) for m in mentors]
    tfidf_matrix = vectorizer.fit_transform(mentor_tags)


def recommend_mentors(user_input, top_n=5):
    """Return top_n mentors based on problem similarity"""
    input_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return [mentors[i] for i in top_indices]

# ====================== CHATBOT SYSTEM ======================

# Initialize NLP components for chatbot
lemmatizer = WordNetLemmatizer()

# Load chatbot model and data
try:
    chatbot_model = load_model('model.h5')
    intents = json.load(open('intents.json'))
    words = pickle.load(open('texts.pkl', 'rb'))
    classes = pickle.load(open('labels.pkl', 'rb'))
except Exception as e:
    print(f"Error loading chatbot model or data: {e}")
    exit(1)

# Initialize language detection
try:
    nlp = spacy.load("en_core_web_sm")
    
    def get_lang_detector(nlp, name):
        return LanguageDetector()
    
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe('language_detector', last=True)
except Exception as e:
    print(f"Error initializing language detection: {e}")
    exit(1)

# Chatbot helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm not sure how to respond to that. Could you rephrase?"

def chatbot_response(msg):
    try:
        doc = nlp(msg)
        detected_language = doc._.language['language']
        if detected_language != "en":
            return "I currently only support English. Please try again in English."
        return get_response(predict_class(msg, chatbot_model), intents)
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an error processing your message."

# ====================== CHAT ROOM SYSTEM ======================

# Store active chat rooms (in production, use database)
active_chats = {}

# ====================== ROUTES ======================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    problem = request.args.get('problem', '').strip().lower()
    
    if not problem:
        return jsonify({"mentors": mentors[:10]})  # First 10 by default
    
    try:
        results = recommend_mentors(problem)
        return jsonify({"mentors": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/create_chat_room', methods=['POST'])
def create_chat_room():
    """Create a private chat room between user and mentor"""
    user_id = request.json.get('user_id')
    mentor_id = request.json.get('mentor_id')
    room_id = f"chat_{user_id}_{mentor_id}"
    
    if room_id not in active_chats:
        active_chats[room_id] = {
            "participants": {"user": user_id, "mentor": mentor_id},
            "messages": []
        }
    return jsonify({"room_id": room_id})

@app.route('/api/mentor/active_chats', methods=['GET'])
def get_active_chats():
    """Get all active chats for a mentor"""
    mentor_id = request.args.get('mentor_id')
    return jsonify({
        "chats": [room for room in active_chats if f"chat_" in room and mentor_id in room]
    })

@app.route("/get", methods=['POST'])
def get_bot_response():
    try:
        data = request.get_json()
        user_message = data.get('msg', '')
        response = chatbot_response(user_message)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error handling request: {e}")
        return jsonify({"response": "Sorry, I encountered an error."})

# ====================== SOCKET.IO HANDLERS ======================

@socketio.on('join_chat')
def handle_join(data):
    """Handle user/mentor joining a chat room"""
    room_id = data['room_id']
    join_room(room_id)
    emit('chat_message', {
        "sender": "system",
        "text": f"{data['username']} joined the chat",
        "time": datetime.now().strftime("%H:%M")
    }, room=room_id)

@socketio.on('mentor_join')
def handle_mentor_join(data):
    """When mentor joins the dashboard"""
    mentor_id = data['mentor_id']
    join_room(f"mentor_{mentor_id}")  # Special room for mentor notifications
    emit('mentor_status', {"status": f"Mentor {mentor_id} online"})

@socketio.on('send_chat_message')
def handle_message(data):
    """Handle real-time messaging"""
    room_id = data['room_id']
    is_mentor = data['sender'].startswith('mentor_')
    
    message = {
        "sender": data['sender'],
        "text": data['text'],
        "time": datetime.now().strftime("%H:%M"),
        "type": "mentor" if is_mentor else "user"
    }
    
    # Store message (in production, save to database)
    if room_id in active_chats:
        active_chats[room_id]['messages'].append(message)
    
    # Broadcast with sender type info
    emit('chat_message', {
        **message,
        "is_mentor": is_mentor  # Frontend will use this for styling
    }, room=room_id)

# ====================== INITIALIZATION ======================

if __name__ == '__main__':
    # Download NLTK data
    nltk.download('punkt')
    nltk.download('wordnet')
    
    # Run the application
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
# from flask import Flask, jsonify, request, render_template
# from flask_cors import CORS
# from flask_socketio import SocketIO, join_room, emit
# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import nltk
# from nltk.stem import WordNetLemmatizer
# import pickle
# import numpy as np
# from keras.models import load_model
# import random
# import spacy
# from spacy.language import Language
# from spacy_langdetect import LanguageDetector

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend-backend communication

# # Initialize Socket.io
# socketio = SocketIO(app, cors_allowed_origins="*")

# # ====================== MENTOR MATCHING SYSTEM ======================

# # Load mentor data
# with open('mentors.json', 'r') as f:
#     mentors = json.load(f)

# # Prepare ML model for mentor recommendations
# vectorizer = TfidfVectorizer()
# mentor_tags = [' '.join(m['problem_tags']) for m in mentors]
# tfidf_matrix = vectorizer.fit_transform(mentor_tags)

# def recommend_mentors(user_input, top_n=5):
#     """Return top_n mentors based on problem similarity"""
#     input_vec = vectorizer.transform([user_input])
#     sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
#     top_indices = sim_scores.argsort()[-top_n:][::-1]
#     return [mentors[i] for i in top_indices]

# # ====================== CHATBOT SYSTEM ======================

# # Initialize NLP components for chatbot
# lemmatizer = WordNetLemmatizer()

# # Load chatbot model and data
# try:
#     chatbot_model = load_model('model.h5')
#     intents = json.load(open('intents.json'))
#     words = pickle.load(open('texts.pkl', 'rb'))
#     classes = pickle.load(open('labels.pkl', 'rb'))
# except Exception as e:
#     print(f"Error loading chatbot model or data: {e}")
#     exit(1)

# # Initialize language detection
# try:
#     nlp = spacy.load("en_core_web_sm")
    
#     def get_lang_detector(nlp, name):
#         return LanguageDetector()
    
#     Language.factory("language_detector", func=get_lang_detector)
#     nlp.add_pipe('language_detector', last=True)
# except Exception as e:
#     print(f"Error initializing language detection: {e}")
#     exit(1)

# # Chatbot helper functions
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bow(sentence, words, show_details=False):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#                 if show_details:
#                     print(f"found in bag: {w}")
#     return np.array(bag)

# def predict_class(sentence, model):
#     p = bow(sentence, words)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]#po

# def get_response(ints, intents_json):
#     if ints:
#         tag = ints[0]['intent']
#         for i in intents_json['intents']:
#             if i['tag'] == tag:
#                 return random.choice(i['responses'])
#     return "I'm not sure how to respond to that. Could you rephrase?"

# def chatbot_response(msg):
#     try:
#         doc = nlp(msg)
#         detected_language = doc._.language['language']
#         if detected_language != "en":
#             return "I currently only support English. Please try again in English."
#         return get_response(predict_class(msg, chatbot_model), intents)
#     except Exception as e:
#         print(f"Error generating response: {e}")
#         return "Sorry, I encountered an error processing your message."

# # ====================== CHAT ROOM SYSTEM ======================

# # Store active chat rooms (in production, use database)
# active_chats = {}

# # ====================== ROUTES ======================

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/api/recommend', methods=['GET'])
# def api_recommend():
#     problem = request.args.get('problem', '').strip().lower()
    
#     if not problem:
#         return jsonify({"mentors": mentors[:10]})  # First 10 by default
    
#     try:
#         results = recommend_mentors(problem)
#         return jsonify({"mentors": results})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/api/create_chat_room', methods=['POST'])
# def create_chat_room():
#     """Create a private chat room between user and mentor"""
#     user_id = request.json.get('user_id')
#     mentor_id = request.json.get('mentor_id')
#     room_id = f"chat_{user_id}_{mentor_id}"
    
#     if room_id not in active_chats:
#         active_chats[room_id] = {
#             "participants": {"user": user_id, "mentor": mentor_id},
#             "messages": []
#         }
#     return jsonify({"room_id": room_id})

# @app.route('/api/mentor/active_chats', methods=['GET'])
# def get_active_chats():
#     """Get all active chats for a mentor"""
#     mentor_id = request.args.get('mentor_id')
#     return jsonify({
#         "chats": [room for room in active_chats if f"chat_" in room and mentor_id in room]
#     })

# @app.route("/get", methods=['POST'])
# def get_bot_response():
#     try:
#         data = request.get_json()
#         user_message = data.get('msg', '')
#         response = chatbot_response(user_message)
#         return jsonify({"response": response})
#     except Exception as e:
#         print(f"Error handling request: {e}")
#         return jsonify({"response": "Sorry, I encountered an error."})

# # ====================== SOCKET.IO HANDLERS ======================

# @socketio.on('join_chat')
# def handle_join(data):
#     """Handle user/mentor joining a chat room"""
#     room_id = data['room_id']
#     join_room(room_id)
#     emit('chat_message', {
#         "sender": "system",
#         "text": f"{data['username']} joined the chat",
#         "time": datetime.now().strftime("%H:%M")
#     }, room=room_id)

# @socketio.on('mentor_join')
# def handle_mentor_join(data):
#     """When mentor joins the dashboard"""
#     mentor_id = data['mentor_id']
#     join_room(f"mentor_{mentor_id}")  # Special room for mentor notifications
#     emit('mentor_status', {"status": f"Mentor {mentor_id} online"})

# @socketio.on('send_chat_message')
# def handle_message(data):
#     """Handle real-time messaging"""
#     room_id = data['room_id']
#     is_mentor = data['sender'].startswith('mentor_')
    
#     message = {
#         "sender": data['sender'],
#         "text": data['text'],
#         "time": datetime.now().strftime("%H:%M"),
#         "type": "mentor" if is_mentor else "user"
#     }
    
#     # Store message (in production, save to database)
#     if room_id in active_chats:
#         active_chats[room_id]['messages'].append(message)
    
#     # Broadcast with sender type info
#     emit('chat_message', {
#         **message,
#         "is_mentor": is_mentor  # Frontend will use this for styling
#     }, room=room_id)

# # ====================== INITIALIZATION ======================

# if __name__ == '__main__':
#     # Download NLTK data
#     nltk.download('punkt')
#     nltk.download('wordnet')
    
#     # Run the application
#     socketio.run(app, debug=True, port=5002, allow_unsafe_werkzeug=True)