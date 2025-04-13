# from flask_cors import CORS
# from flask import Flask, render_template

# from flask_socketio import SocketIO, join_room, emit  # NEW: Socket.io imports
# import json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime  # NEW: For chat timestamps

# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend-backend communication

# # NEW: Initialize Socket.io
# socketio = SocketIO(app, cors_allowed_origins="*")

# # Load mentor data
# with open('mentors.json', 'r') as f:
#     mentors = json.load(f)

# # Prepare ML model (Original recommendation system)
# vectorizer = TfidfVectorizer()
# mentor_tags = [' '.join(m['problem_tags']) for m in mentors]
# tfidf_matrix = vectorizer.fit_transform(mentor_tags)

# def recommend_mentors(user_input, top_n=5):
#     """Return top_n mentors based on problem similarity"""
#     input_vec = vectorizer.transform([user_input])
#     sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
#     top_indices = sim_scores.argsort()[-top_n:][::-1]
#     return [mentors[i] for i in top_indices]

# # ====================== ORIGINAL CODE (UNCHANGED) ======================
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

# @app.route('/')
# def home():
#     return render_template('index.html')
# # =======================================================================

# # ====================== NEW CHAT FUNCTIONALITY =========================
# # Store active chat rooms (in production, use database)
# active_chats = {}

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

# @socketio.on('send_chat_message')
# def handle_message(data):
#     """Handle real-time messaging"""
#     room_id = data['room_id']
#     message = {
#         "sender": data['sender'],
#         "text": data['text'],
#         "time": datetime.now().strftime("%H:%M")
#     }
    
#     # Store message (in production, save to database)
#     active_chats[room_id]['messages'].append(message)
    
#     # Broadcast to room
#     emit('chat_message', message, room=room_id)
# # =======================================================================
#     # ====================== NEW MENTOR CHAT FUNCTIONALITY ==================
# @app.route('/api/mentor/active_chats', methods=['GET'])
# def get_active_chats():
#     """Get all active chats for a mentor (NEW)"""
#     mentor_id = request.args.get('mentor_id')
#     return jsonify({
#         "chats": [room for room in active_chats if f"chat_" in room and mentor_id in room]
#     })

# @socketio.on('mentor_join')
# def handle_mentor_join(data):
#     """When mentor joins the dashboard (NEW)"""
#     mentor_id = data['mentor_id']
#     join_room(f"mentor_{mentor_id}")  # Special room for mentor notifications
#     emit('mentor_status', {"status": f"Mentor {mentor_id} online"})

# # Modified message handler to distinguish user/mentor messages
# @socketio.on('send_chat_message')
# def handle_message(data):
#     """Handle real-time messaging (UPDATED)"""
#     room_id = data['room_id']
#     is_mentor = data['sender'].startswith('mentor_')
    
#     message = {
#         "sender": data['sender'],
#         "text": data['text'],
#         "time": datetime.now().strftime("%H:%M"),
#         "type": "mentor" if is_mentor else "user"  # NEW: Identify sender type
#     }
    
#     active_chats[room_id]['messages'].append(message)
    
#     # Broadcast with sender type info (NEW)
#     emit('chat_message', {
#         **message,
#         "is_mentor": is_mentor  # Frontend will use this for styling
#     }, room=room_id)
# # =======================================================================


# if __name__ == '__main__':
#     socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)  
# import json
# import pandas as pd
# import numpy as np
# from flask import Flask, request, jsonify
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import NearestNeighbors
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# from nltk.stem import WordNetLemmatizer
# import joblib
# import re
# from concurrent.futures import ThreadPoolExecutor

# # --- Advanced Text Preprocessing ---
# class AdvancedTextPreprocessor:
#     def __init__(self):
#         self.lemmatizer = WordNetLemmatizer()
#         self.stop_words = set(ENGLISH_STOP_WORDS)
        
#     def preprocess(self, text):
#         # Lowercase
#         text = text.lower()
#         # Remove special chars
#         text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#         # Tokenize and lemmatize
#         tokens = [self.lemmatizer.lemmatize(word) for word in text.split() 
#                  if word not in self.stop_words]
#         return ' '.join(tokens)

# # --- Hybrid Recommender System ---
# class HybridRecommender:
#     def __init__(self):
#         self.model = Pipeline([
#             ('tfidf', TfidfVectorizer(
#                 token_pattern=r'(?u)\b[A-Za-z]+\b',
#                 stop_words='english',
#                 ngram_range=(1, 2)
#             )),
#             ('nn', NearestNeighbors(n_neighbors=5, metric='cosine'))
#         ])
#         self.mentors_df = None
#         self.rating_weights = None
        
#     def fit(self, mentors_data):
#         # Create enhanced features
#         self.mentors_df = pd.DataFrame(mentors_data)
#         self.mentors_df['combined_features'] = self.mentors_df.apply(
#             lambda x: f"{' '.join(x['issues_helped'])} {x['experience']}", axis=1)
        
#         # Preprocess text
#         preprocessor = AdvancedTextPreprocessor()
#         self.mentors_df['processed_features'] = self.mentors_df['combined_features'].apply(preprocessor.preprocess)
        
#         # Calculate rating weights (normalized)
#         self.rating_weights = (self.mentors_df['rating'] - self.mentors_df['rating'].min()) / \
#                             (self.mentors_df['rating'].max() - self.mentors_df['rating'].min())
        
#         # Fit model
#         self.model.fit(self.mentors_df['processed_features'])
        
#     def recommend(self, query, communication_pref, top_n=3):
#         # Preprocess query
#         preprocessor = AdvancedTextPreprocessor()
#         processed_query = preprocessor.preprocess(query)
        
#         # Find similar mentors
#         query_vec = self.model.named_steps['tfidf'].transform([processed_query])
#         distances, indices = self.model.named_steps['nn'].kneighbors(query_vec)
        
#         # Filter by communication preference and apply rating weights
#         results = []
#         for i, idx in enumerate(indices[0]):
#             mentor = self.mentors_df.iloc[idx].to_dict()
#             if mentor['communication'] == communication_pref:
#                 # Combine content similarity (60%) and rating (40%)
#                 score = 0.6 * (1 - distances[0][i]) + 0.4 * self.rating_weights[idx]
#                 mentor['match_score'] = round(score, 2)
#                 results.append(mentor)
        
#         # Sort and return top results
#         return sorted(results, key=lambda x: x['match_score'], reverse=True)[:top_n]

# # --- Flask Application ---
# app = Flask(__name__)
# recommender = HybridRecommender()
# executor = ThreadPoolExecutor(2)

# # Load data and initialize model
# with open("mentors.json", "r") as f:
#     mentors_data = json.load(f)
#     recommender.fit(mentors_data)

# @app.route("/match", methods=["GET"])
# def match_mentor():
#     try:
#         user_problem = request.args.get("problem", "").strip()
#         communication = request.args.get("communication", "text").strip().lower()
        
#         if not user_problem:
#             return jsonify({"error": "Problem parameter is required"}), 400
            
#         # Get recommendations (async for scalability)
#         future = executor.submit(recommender.recommend, user_problem, communication)
#         results = future.result()
        
#         if not results:
#             return jsonify({"message": "No mentors found matching your criteria", "suggestions": [
#                 "Try broadening your search terms",
#                 "Check back later as we add more mentors"
#             ]}), 404
            
#         # Prepare response
#         response = {
#             "top_matches": [{
#                 "id": m["id"],
#                 "name": m["name"],
#                 "match_score": m["match_score"],
#                 "issues_helped": m["issues_helped"],
#                 "experience": m["experience"],
#                 "rating": m["rating"]
#             } for m in results],
#             "search_analysis": {
#                 "processed_query": user_problem,
#                 "total_possible_matches": len(results)
#             }
#         }
#         return jsonify(response)
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, threaded=True)
# import json
# import pandas as pd
# from flask import Flask, request, jsonify
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# from nltk.stem import WordNetLemmatizer
# import re
# from concurrent.futures import ThreadPoolExecutor
# import os
# from pathlib import Path

# # --- Advanced Text Preprocessing ---
# class AdvancedTextPreprocessor:
#     def __init__(self):
#         self.lemmatizer = WordNetLemmatizer()
#         self.stop_words = set(ENGLISH_STOP_WORDS)
        
#     def preprocess(self, text):
#         text = text.lower()
#         text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#         tokens = [self.lemmatizer.lemmatize(word) for word in text.split() 
#                  if word not in self.stop_words]
#         return ' '.join(tokens)

# # --- Hybrid Recommender System ---
# class HybridRecommender:
#     def __init__(self):
#         self.model = Pipeline([
#             ('tfidf', TfidfVectorizer(
#                 token_pattern=r'(?u)\b[A-Za-z]+\b',
#                 stop_words='english',
#                 ngram_range=(1, 2)
#             )),
#             ('nn', NearestNeighbors(n_neighbors=5, metric='cosine'))
#         ])
#         self.mentors_df = None
#         self.rating_weights = None
        
#     def fit(self, mentors_data):
#         self.mentors_df = pd.DataFrame(mentors_data)
        
#         # Verify required columns exist
#         required_columns = ['problem_tags', 'name', 'language', 'rating']
#         for col in required_columns:
#             if col not in self.mentors_df.columns:
#                 raise ValueError(f"Missing required column in mentors.json: {col}")
        
#         self.mentors_df['combined_features'] = self.mentors_df.apply(
#             lambda x: f"{' '.join(x['problem_tags'])} {x['rating']}", axis=1)
        
#         preprocessor = AdvancedTextPreprocessor()
#         self.mentors_df['processed_features'] = self.mentors_df['combined_features'].apply(preprocessor.preprocess)
        
#         self.rating_weights = (self.mentors_df['rating'] - self.mentors_df['rating'].min()) / \
#                             (self.mentors_df['rating'].max() - self.mentors_df['rating'].min())
        
#         self.model.fit(self.mentors_df['processed_features'])
        
#     def recommend(self, query, communication_pref, top_n=3):
#         preprocessor = AdvancedTextPreprocessor()
#         processed_query = preprocessor.preprocess(query)
        
#         query_vec = self.model.named_steps['tfidf'].transform([processed_query])
#         distances, indices = self.model.named_steps['nn'].kneighbors(query_vec)
        
#         results = []
#         for i, idx in enumerate(indices[0]):
#             mentor = self.mentors_df.iloc[idx].to_dict()
#             if mentor['communication'].lower() == communication_pref.lower():
#                 score = 0.6 * (1 - distances[0][i]) + 0.4 * self.rating_weights[idx]
#                 mentor['match_score'] = round(score, 2)
#                 results.append(mentor)
        
#         return sorted(results, key=lambda x: x['match_score'], reverse=True)[:top_n]

# # --- Flask Application ---
# app = Flask(__name__)
# recommender = HybridRecommender()
# executor = ThreadPoolExecutor(2)

# # Load data from existing mentors.json
# try:
#     with open("mentors.json", "r") as f:
#         mentors_data = json.load(f)
#     recommender.fit(mentors_data)
#     print("✅ Successfully loaded mentors.json and initialized model")
# except Exception as e:
#     print(f"❌ Failed to initialize: {str(e)}")
#     raise

# @app.route("/")
# def home():
#     return jsonify({
#         "status": "active",
#         "endpoints": {
#             "/match": "GET ?problem=<issue>&communication=<type>"
#         },
#         "message": "Mentor Matching API is running"
#     })

# @app.route("/match", methods=["GET"])
# def match_mentor():
#     try:
#         user_problem = request.args.get("problem", "").strip()
#         communication = request.args.get("communication", "text").strip().lower()
        
#         if not user_problem:
#             return jsonify({"error": "Please provide a 'problem' parameter"}), 400
            
#         results = recommender.recommend(user_problem, communication)
        
#         if not results:
#             return jsonify({
#                 "message": "No matching mentors found",
#                 "suggestions": [
#                     "Try different keywords like 'stress' or 'anxiety'",
#                     "Check if your communication preference is available"
#                 ]
#             }), 404
            
#         return jsonify({
#             "matches": results,
#             "query": user_problem,
#             "count": len(results)
#         })
        
#     except Exception as e:
#         return jsonify({
#             "error": "Search failed",
#             "details": str(e)
#         }), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001, debug=True)
# import json
# import pandas as pd
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from flask_socketio import SocketIO, join_room, emit
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# from nltk.stem import WordNetLemmatizer
# import re
# from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime
# import os
# from pathlib import Path

# # Initialize Flask app with CORS and SocketIO
# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*")

# # --- Advanced Text Preprocessing ---
# class AdvancedTextPreprocessor:
#     def __init__(self):
#         self.lemmatizer = WordNetLemmatizer()
#         self.stop_words = set(ENGLISH_STOP_WORDS)
        
#     def preprocess(self, text):
#         text = text.lower()
#         text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#         tokens = [self.lemmatizer.lemmatize(word) for word in text.split() 
#                  if word not in self.stop_words]
#         return ' '.join(tokens)

# # --- Hybrid Recommender System ---
# class HybridRecommender:
#     def __init__(self):
#         self.model = Pipeline([
#             ('tfidf', TfidfVectorizer(
#                 token_pattern=r'(?u)\b[A-Za-z]+\b',
#                 stop_words='english',
#                 ngram_range=(1, 2)
#             )),
#             ('nn', NearestNeighbors(n_neighbors=5, metric='cosine'))
#         ])
#         self.mentors_df = None
#         self.rating_weights = None
        
#     def fit(self, mentors_data):
#         self.mentors_df = pd.DataFrame(mentors_data)
        
#         # Verify required columns exist
#         required_columns = ['problem_tags', 'name', 'language', 'rating', 'id']
#         for col in required_columns:
#             if col not in self.mentors_df.columns:
#                 raise ValueError(f"Missing required column in mentors.json: {col}")
        
#         self.mentors_df['combined_features'] = self.mentors_df.apply(
#             lambda x: f"{' '.join(x['problem_tags'])} {x['rating']}", axis=1)
        
#         preprocessor = AdvancedTextPreprocessor()
#         self.mentors_df['processed_features'] = self.mentors_df['combined_features'].apply(preprocessor.preprocess)
        
#         self.rating_weights = (self.mentors_df['rating'] - self.mentors_df['rating'].min()) / \
#                             (self.mentors_df['rating'].max() - self.mentors_df['rating'].min())
        
#         self.model.fit(self.mentors_df['processed_features'])
        
#     def recommend(self, query, communication_pref, top_n=3):
#         preprocessor = AdvancedTextPreprocessor()
#         processed_query = preprocessor.preprocess(query)
        
#         query_vec = self.model.named_steps['tfidf'].transform([processed_query])
#         distances, indices = self.model.named_steps['nn'].kneighbors(query_vec)
        
#         results = []
#         for i, idx in enumerate(indices[0]):
#             mentor = self.mentors_df.iloc[idx].to_dict()
#             if mentor['communication'].lower() == communication_pref.lower():
#                 score = 0.6 * (1 - distances[0][i]) + 0.4 * self.rating_weights[idx]
#                 mentor['match_score'] = round(score, 2)
#                 results.append(mentor)
        
#         return sorted(results, key=lambda x: x['match_score'], reverse=True)[:top_n]

# # Initialize recommender system
# recommender = HybridRecommender()
# executor = ThreadPoolExecutor(2)

# # Store active chat rooms (in production, use database)
# active_chats = {}

# # Load data from existing mentors.json
# try:
#     with open("mentors.json", "r") as f:
#         mentors_data = json.load(f)
#     recommender.fit(mentors_data)
#     print("✅ Successfully loaded mentors.json and initialized model")
# except Exception as e:
#     print(f"❌ Failed to initialize: {str(e)}")
#     raise

# # ====================== ROUTES ======================
# @app.route("/")
# def home():
#     return render_template('index.html')

# @app.route("/api/status")
# def status():
#     return jsonify({
#         "status": "active",
#         "endpoints": {
#             "/api/match": "GET ?problem=<issue>&communication=<type>",
#             "/api/create_chat_room": "POST with user_id and mentor_id",
#             "/api/mentor/active_chats": "GET ?mentor_id=<id>"
#         },
#         "message": "Mentor Matching and Chat API is running"
#     })

# @app.route("/api/match", methods=["GET"])
# def match_mentor():
#     try:
#         user_problem = request.args.get("problem", "").strip()
#         communication = request.args.get("communication", "text").strip().lower()
        
#         if not user_problem:
#             return jsonify({"error": "Please provide a 'problem' parameter"}), 400
            
#         results = recommender.recommend(user_problem, communication)
        
#         if not results:
#             return jsonify({
#                 "message": "No matching mentors found",
#                 "suggestions": [
#                     "Try different keywords like 'stress' or 'anxiety'",
#                     "Check if your communication preference is available"
#                 ]
#             }), 404
            
#         return jsonify({
#             "matches": results,
#             "query": user_problem,
#             "count": len(results)
#         })
        
#     except Exception as e:
#         return jsonify({
#             "error": "Search failed",
#             "details": str(e)
#         }), 500

# @app.route('/api/create_chat_room', methods=['POST'])
# def create_chat_room():
#     """Create a private chat room between user and mentor"""
#     user_id = request.json.get('user_id')
#     mentor_id = request.json.get('mentor_id')
#     room_id = f"chat_{user_id}_{mentor_id}"
    
#     if room_id not in active_chats:
#         active_chats[room_id] = {
#             "participants": {"user": user_id, "mentor": mentor_id},
#             "messages": [],
#             "mentor_name": next((m['name'] for m in mentors_data if str(m['id']) == str(mentor_id)), "Mentor"),
#             "user_name": f"User_{user_id}"
#         }
#     return jsonify({
#         "room_id": room_id,
#         "mentor_name": active_chats[room_id]['mentor_name']
#     })

# @app.route('/api/mentor/active_chats', methods=['GET'])
# def get_active_chats():
#     """Get all active chats for a mentor"""
#     mentor_id = request.args.get('mentor_id')
#     mentor_chats = []
    
#     for room_id, chat_data in active_chats.items():
#         if f"chat_" in room_id and str(mentor_id) in room_id:
#             mentor_chats.append({
#                 "room_id": room_id,
#                 "user_id": chat_data['participants']['user'],
#                 "user_name": chat_data['user_name'],
#                 "last_message": chat_data['messages'][-1]['text'] if chat_data['messages'] else None,
#                 "unread_count": sum(1 for msg in chat_data['messages'] if msg['type'] == 'user')
#             })
    
#     return jsonify({"chats": mentor_chats})

# # ====================== SOCKET.IO HANDLERS ======================
# @socketio.on('join_chat')
# def handle_join(data):
#     """Handle user/mentor joining a chat room"""
#     room_id = data['room_id']
#     join_room(room_id)
    
#     # Update user/mentor name if provided
#     if 'username' in data:
#         if data['sender_type'] == 'user':
#             active_chats[room_id]['user_name'] = data['username']
#         else:
#             active_chats[room_id]['mentor_name'] = data['username']
    
#     emit('chat_message', {
#         "sender": "system",
#         "text": f"{data.get('username', 'Someone')} joined the chat",
#         "time": datetime.now().strftime("%H:%M"),
#         "type": "system"
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
#     is_mentor = data.get('sender_type', 'user') == 'mentor'
    
#     message = {
#         "sender": data['sender_id'],
#         "text": data['text'],
#         "time": datetime.now().strftime("%H:%M"),
#         "type": "mentor" if is_mentor else "user"
#     }
    
#     # Store message
#     if room_id in active_chats:
#         active_chats[room_id]['messages'].append(message)
    
#     # Broadcast to room
#     emit('chat_message', {
#         **message,
#         "sender_name": active_chats[room_id]['mentor_name'] if is_mentor else active_chats[room_id]['user_name'],
#         "is_mentor": is_mentor
#     }, room=room_id)
    
#     # Notify mentor if they're not in the chat
#     if not is_mentor:
#         mentor_id = active_chats[room_id]['participants']['mentor']
#         emit('new_message', {
#             "room_id": room_id,
#             "message": message['text'],
#             "sender": active_chats[room_id]['user_name']
#         }, room=f"mentor_{mentor_id}")

# if __name__ == '__main__':
#     socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, emit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path

# Initialize Flask app with CORS and SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Advanced Text Preprocessing ---
class AdvancedTextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(ENGLISH_STOP_WORDS)
        
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = [self.lemmatizer.lemmatize(word) for word in text.split() 
                 if word not in self.stop_words]
        return ' '.join(tokens)

# --- Hybrid Recommender System ---
class HybridRecommender:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                token_pattern=r'(?u)\b[A-Za-z]+\b',
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('nn', NearestNeighbors(n_neighbors=5, metric='cosine'))
        ])
        self.mentors_df = None
        self.rating_weights = None
        
    def fit(self, mentors_data):
        self.mentors_df = pd.DataFrame(mentors_data)
        
        # Verify required columns exist
        required_columns = ['id', 'name', 'problem_tags', 'language', 'rating']
        for col in required_columns:
            if col not in self.mentors_df.columns:
                raise ValueError(f"Missing required column in mentors.json: {col}")
        
        self.mentors_df['combined_features'] = self.mentors_df.apply(
            lambda x: f"{' '.join(x['problem_tags'])} {x['rating']}", axis=1)
        
        preprocessor = AdvancedTextPreprocessor()
        self.mentors_df['processed_features'] = self.mentors_df['combined_features'].apply(preprocessor.preprocess)
        
        self.rating_weights = (self.mentors_df['rating'] - self.mentors_df['rating'].min()) / \
                            (self.mentors_df['rating'].max() - self.mentors_df['rating'].min())
        
        self.model.fit(self.mentors_df['processed_features'])
        
    def recommend(self, query, top_n=3):
        preprocessor = AdvancedTextPreprocessor()
        processed_query = preprocessor.preprocess(query)
        
        query_vec = self.model.named_steps['tfidf'].transform([processed_query])
        distances, indices = self.model.named_steps['nn'].kneighbors(query_vec)
        
        results = []
        for i, idx in enumerate(indices[0]):
            mentor = self.mentors_df.iloc[idx].to_dict()
            score = 0.6 * (1 - distances[0][i]) + 0.4 * self.rating_weights[idx]
            mentor['match_score'] = round(score, 2)
            results.append(mentor)
        
        return sorted(results, key=lambda x: x['match_score'], reverse=True)[:top_n]

# Initialize recommender system
recommender = HybridRecommender()
executor = ThreadPoolExecutor(2)

# Store active chat rooms (in production, use database)
active_chats = {}

# Load data from existing mentors.json
try:
    with open("mentors.json", "r") as f:
        mentors_data = json.load(f)
    recommender.fit(mentors_data)
    print("✅ Successfully loaded mentors.json and initialized model")
except Exception as e:
    print(f"❌ Failed to initialize: {str(e)}")
    raise

# ====================== ROUTES ======================
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/api/status")
def status():
    return jsonify({
        "status": "active",
        "endpoints": {
            "/api/match": "GET ?problem=<issue>",
            "/api/create_chat_room": "POST with user_id and mentor_id",
            "/api/mentor/active_chats": "GET ?mentor_id=<id>"
        },
        "message": "Text-Based Mentor Matching and Chat API is running"
    })

@app.route("/api/match", methods=["GET"])
def match_mentor():
    try:
        user_problem = request.args.get("problem", "").strip()
        
        if not user_problem:
            return jsonify({"error": "Please provide a 'problem' parameter"}), 400
            
        results = recommender.recommend(user_problem)
        
        if not results:
            return jsonify({
                "message": "No matching mentors found",
                "suggestions": [
                    "Try different keywords like 'stress' or 'anxiety'"
                ]
            }), 404
            
        return jsonify({
            "matches": results,
            "query": user_problem,
            "count": len(results)
        })
        
    except Exception as e:
        return jsonify({
            "error": "Search failed",
            "details": str(e)
        }), 500

@app.route('/api/create_chat_room', methods=['POST'])
def create_chat_room():
    """Create a private chat room between user and mentor"""
    user_id = request.json.get('user_id')
    mentor_id = request.json.get('mentor_id')
    room_id = f"chat_{user_id}_{mentor_id}"
    
    if room_id not in active_chats:
        mentor = next((m for m in mentors_data if str(m['id']) == str(mentor_id)), None)
        if not mentor:
            return jsonify({"error": "Mentor not found"}), 404
            
        active_chats[room_id] = {
            "participants": {"user": user_id, "mentor": mentor_id},
            "messages": [],
            "mentor_name": mentor.get('name', 'Mentor'),
            "user_name": f"User_{user_id}",
            "created_at": datetime.now().isoformat()
        }
    return jsonify({
        "room_id": room_id,
        "mentor_name": active_chats[room_id]['mentor_name']
    })

@app.route('/api/mentor/active_chats', methods=['GET'])
def get_active_chats():
    """Get all active chats for a mentor"""
    mentor_id = request.args.get('mentor_id')
    if not mentor_id:
        return jsonify({"error": "Missing mentor_id parameter"}), 400
        
    mentor_chats = []
    
    for room_id, chat_data in active_chats.items():
        if f"_{mentor_id}" in room_id:
            last_message = chat_data['messages'][-1] if chat_data['messages'] else None
            mentor_chats.append({
                "room_id": room_id,
                "user_id": chat_data['participants']['user'],
                "user_name": chat_data['user_name'],
                "last_message": last_message['text'] if last_message else None,
                "last_message_time": last_message['time'] if last_message else None,
                "unread_count": sum(1 for msg in chat_data['messages'] if msg.get('type') == 'user'),
                "created_at": chat_data['created_at']
            })
    
    return jsonify({"chats": sorted(mentor_chats, key=lambda x: x['last_message_time'] or x['created_at'], reverse=True)})

# ====================== SOCKET.IO HANDLERS ======================
@socketio.on('join_chat')
def handle_join(data):
    """Handle user/mentor joining a chat room"""
    room_id = data.get('room_id')
    if not room_id or room_id not in active_chats:
        emit('join_error', {"message": "Invalid room ID"})
        return
        
    join_room(room_id)
    
    # Update participant name if provided
    if 'username' in data and 'sender_type' in data:
        if data['sender_type'] == 'user':
            active_chats[room_id]['user_name'] = data['username']
        else:
            active_chats[room_id]['mentor_name'] = data['username']
    
    # Send join notification
    emit('chat_message', {
        "sender": "system",
        "text": f"{data.get('username', 'Someone')} joined the chat",
        "time": datetime.now().strftime("%H:%M"),
        "type": "system"
    }, room=room_id)
    
    # Send chat history to the new participant
    if active_chats[room_id]['messages']:
        emit('chat_history', {
            "messages": active_chats[room_id]['messages']
        })

@socketio.on('mentor_join')
def handle_mentor_join(data):
    """When mentor joins the dashboard"""
    mentor_id = data.get('mentor_id')
    if not mentor_id:
        return
        
    join_room(f"mentor_{mentor_id}")
    emit('mentor_status', {"status": "online", "mentor_id": mentor_id})

@socketio.on('send_chat_message')
def handle_message(data):
    """Handle real-time messaging"""
    room_id = data.get('room_id')
    if not room_id or room_id not in active_chats:
        emit('message_error', {"message": "Invalid room ID"})
        return
        
    is_mentor = data.get('sender_type') == 'mentor'
    sender_id = data.get('sender_id')
    message_text = data.get('text', '').strip()
    
    if not sender_id or not message_text:
        return
        
    message = {
        "sender": sender_id,
        "text": message_text,
        "time": datetime.now().strftime("%H:%M"),
        "type": "mentor" if is_mentor else "user",
        "sender_name": active_chats[room_id]['mentor_name'] if is_mentor else active_chats[room_id]['user_name']
    }
    
    # Store message
    active_chats[room_id]['messages'].append(message)
    
    # Broadcast to room
    emit('chat_message', message, room=room_id)
    
    # Notify mentor if message is from user
    if not is_mentor:
        mentor_id = active_chats[room_id]['participants']['mentor']
        emit('new_message_notification', {
            "room_id": room_id,
            "message": message_text,
            "sender": active_chats[room_id]['user_name'],
            "time": message['time']
        }, room=f"mentor_{mentor_id}")

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5001, debug=True, allow_unsafe_werkzeug=True)