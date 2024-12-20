from flask import Flask, render_template, request, jsonify
from new import SentimentPipeline  # Import your back-end script

app = Flask(__name__)

# Initialize sentiment pipeline
sentiment_pipeline = SentimentPipeline(use_vader=True)  # Adjust based on your choice

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html is in the templates/ folder

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    sentiment = sentiment_pipeline.process_input(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
