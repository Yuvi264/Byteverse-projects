# 🧠 AnonMind - AI-Powered Mental Health Support Platform

> **Empowering Minds, Anonymously.**  
AnonMind is an AI-driven mental health platform that provides accessible, anonymous, and empathetic support through a blend of chatbot assistance, peer mentorship, and professional therapy 

## 📚 Table of Contents
1. [📝 Description](#-description)  
2. [🌟 Project Vision](#-project-vision)  
3. [🚨 Problem Statement](#-problem-statement)  
4. [🧰 Tech Stack](#-tech-stack)  
5. [💡 Solution Overview](#-solution-overview)  
6. [✨ Features](#-features)  
7. [🔁 Platform Flow](#-platform-flow)  
8. [⚙️ Installation & Setup](#-installation--setup)  
9. [🚀 Usage](#-usage)  
10. [🎥 Demo](#-demo)

---

## 📝 Description

AnonMind connects users anonymously with peers who’ve overcome similar challenges and with licensed therapists for professional care. Our platform uses AI for emotional analysis, smart matching, and NLP responses.

---

## 🌟 Project Vision

Mental health care should be:
- **Affordable**
- **Relatable**
- **Anonymous**
- **Empathetic**

> AnonMind envisions a world where support is available without stigma, delays, or high costs — combining AI tools with human guidance.

---

## 🚨 Problem Statement

1. 💸 Traditional therapy is expensive (avg. $100–$200/session).  
2. 😶 Stigma prevents people from opening up to professionals.  
3. 🤝 Existing platforms lack anonymous, peer-driven guidance.  
4. 🔐 Centralized systems pose privacy and trust issues.

---

## 🧰 Tech Stack

| Layer       | Tech Used                                    |
|------------|---------------------------------------------- |
| Frontend    | HTML, CSS, JavaScript, Tailwind CSS          |
| Backend     |  Flask                                       |
| AI/ML       | TensorFlow, Hugging Face (NLP), Scikit-learn |
| Storage     | Mysql, JSON                                  |
| APIs        | socketio                                    |

---

## 💡 Solution Overview

AnonMind is a **decentralized mental health platform** that:
1. 🧠 Uses an AI chatbot for emotional analysis and initial support.  
2. 🤝 Matches users to **peer mentors** with lived experiences.  
3. 👨‍⚕️ Escalates serious cases to **licensed therapists**.  
4. 📊 Connects users with correct mentors using cosine similarity rule in ML
5. 👨‍⚕️Gives real time chat experience using SocketIO.
6. 🤝Separate dashboard for user and mentors 
7. 🌏Maintaining user authenticity by using sqlalchamy for using user information



---

## ✨ Features

### 🤖 AI-Powered Chatbot
- NLP to detect emotions and mental health conditions (anxiety, stress, etc.)
- Offers coping techniques and suggests next steps

### 🧑‍🤝‍🧑 Peer-to-Peer Mentoring
- Anonymous communication with verified mentors
- Secure chat with AES encryption
- mentors mapping using cosine similarity algorithm

### 🧑‍⚕️ Therapist Integration
- One-click escalation to licensed professionals
- Smooth transition from mentors to therapists

### 🔍 Smart Matching
- ML algorithms (content-based & collaborative filtering)
- Personalizes mentor/therapist recommendations

### 🔁 Feedback Loop
- Collects feedback on sessions to train models and improve accuracy

### 🛡️ Privacy & Security
- End-to-end encryption  
- Decentralized data storage (IPFS)  
- Anonymized user data  

---

## 🔁 Platform Flow

🧩 Step	🔍 Description
1️⃣ User Onboarding	- User lands on web/mobile app.
- Signs up anonymously using email (no personal data).
2️⃣ AI Assessment	- NLP-based chatbot starts conversation.
- Detects issue level:
3️⃣ Smart Matching	- ML algorithms match user with peer mentor or therapist.
- Based on issue type, preference, and availability.
4️⃣ Real-Time Support	- Secure chat with mentor or therapist.
- Uses Socket.IO for real-time communication.
5️⃣ Feedback Loop	- User rates the session.
- Feedback used to improve AI and matching system.

 6️⃣ Data Privacy & Storage	- Data is anonymized and encrypted (AES-256).
- 
Video Link: https://youtu.be/kDtvTpJ-N7g?si=0nR2q9JZN4-GqNds
## Important
🔮 Future Improvements
🌐 Full Deployment

Use platforms like AWS, Azure, or Railway for full deployment support (handle large file sizes).

📱 Mobile App Version

Build a cross-platform mobile app using Flutter or React Native for better accessibility.

🎙️ Voice-Based Interaction

Integrate voice input/output for chatbot using speech-to-text and text-to-speech APIs.

🧠 Better Emotion Detection

Improve NLP models using transformer-based models like BERT or GPT for deeper emotion understanding.

💬 Multi-Language Support

Add regional languages for chatbot and mentor communication to help more users.

📊 Advanced Analytics Dashboard

Add dashboards for therapists and admins to monitor user progress and mentor performance.

🛡️ Blockchain for Data Security

Use blockchain to log session records securely and ensure full transparency.

🔔 Smart Notifications

Implement a smart notification system to remind users about sessions and self-care activities.

🤝 Group Support Sessions

Allow users to join group chats based on their issues and connect with people with similar challenges.

📚 Resource Library

Add mental health articles, videos, and tools for users to learn and practice self-care.


