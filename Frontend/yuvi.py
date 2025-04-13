from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'  # Serve static files from the 'static' folder

# Home route
@app.route("/")
def home():
    return render_template("./templates/index.html")  # Render the main HTML page

# Route to handle chatbot responses
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')  # Get user input from the query parameter 'msg'
    return chatbot_response(userText)  # Return the chatbot's response

# Function to generate chatbot responses (dummy implementation)
def chatbot_response(userText):
    # Replace this with your actual chatbot logic
    if "hello" in userText.lower():
        return "Hello! How can I help you?"
    elif "how are you" in userText.lower():
        return "I'm just a bot, but I'm doing great! How about you?"
    else:
        return "I'm sorry, I didn't understand that."

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)  # Run in debug mode for development