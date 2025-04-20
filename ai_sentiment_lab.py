# Import required libraries
import gradio as gr
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from textblob import TextBlob
from bertopic import BERTopic
import text2emotion as te
from nrclex import NRCLex
import docx
import PyPDF2
import warnings

# Ignore warnings for clean output
warnings.filterwarnings("ignore")

# Initialize required models and analyzers
sentiment_analyzer = SentimentIntensityAnalyzer()  # VADER Sentiment Analyzer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Huggingface summarization pipeline
fake_news_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")  # Fake news detection model
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=5)  # Emotion detection using DistilRoBERTa
topic_model = BERTopic()  # BERTopic for topic modeling

# 📥 Input Handlers
def scrape_website(url):
    # Scrape and return main article or paragraph text from the given URL
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        article = soup.find('article')
        return article.get_text() if article else ' '.join([p.get_text() for p in soup.find_all('p')])
    except Exception as e:
        print(f"❌ Website error: {e}")
        return None

def extract_text_from_pdf(file_path):
    # Extract text content from a PDF file
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
            return text
    except Exception as e:
        print(f"❌ PDF error: {e}")
        return None

def extract_text_from_docx(file_path):
    # Extract text from a DOCX (Word) file
    try:
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"❌ Word file error: {e}")
        return None

# 🧠 Analysis Functions
def get_sentiment(text):
    # Perform sentiment analysis using VADER
    return sentiment_analyzer.polarity_scores(text)

def get_emotions(text):
    # Detect emotions using three models: text2emotion, huggingface, and NRC lexicon
    try:
        emotions_te = te.get_emotion(text)
        emotions_te = {k: round(v, 2) for k, v in emotions_te.items() if v > 0}
        
        emotions_bert = emotion_model(text[:512])
        emotions_hf = {emo['label']: round(emo['score'], 2) for emo in emotions_bert[0] if emo['score'] > 0.05}

        text_object = NRCLex(text)
        nrc_emotions_raw = text_object.raw_emotion_scores
        total_words = sum(nrc_emotions_raw.values()) if nrc_emotions_raw else 1
        nrc_emotions = {k: round(v / total_words, 2) for k, v in nrc_emotions_raw.items() if v > 0}

        return {**emotions_te, **emotions_hf, **nrc_emotions}
    except Exception as e:
        print(f"⚠️ Emotion detection error: {e}")
        return {}

def summarize_text(text):
    # Generate a summary using the BART model
    try:
        if len(text.split()) < 50:
            return text
        summary = summarizer(text[:1024], max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"⚠️ Summary error: {e}")
        return text

def detect_fake_news(text):
    # Predict whether the given text is fake news
    try:
        result = fake_news_classifier(text[:512])
        return result[0]['label'], result[0]['score']
    except Exception:
        return "Error", 0.0

def extract_topics(text):
    # Extract topics using BERTopic
    try:
        topics, _ = topic_model.fit_transform([text])
        return topic_model.get_topic_info()
    except Exception as e:
        print(f"⚠️ Topic modeling error: {e}")
        return None

# 🖼 Display Functions
def display_sentiment(sentiment):
    # Format sentiment results for display
    return f"🧠 Sentiment Analysis:\n🟢 Positive: {sentiment['pos']*100:.2f}%\n🔴 Negative: {sentiment['neg']*100:.2f}%\n🟡 Neutral:  {sentiment['neu']*100:.2f}%\n⚖️ Compound Score: {sentiment['compound']*100:.2f}%"

def display_emotions(emotions):
    # Format emotion detection results
    if not emotions:
        return "❌ No emotions detected."
    result = "🎭 Emotion Detection:\n"
    for emotion, score in sorted(emotions.items(), key=lambda x: -x[1]):
        result += f"{emotion.capitalize()}: {score:.2f}\n"
    return result

def display_topics(topics):
    # Format topic modeling results
    if topics is not None:
        result = "📌 Extracted Topics:\n"
        for i, row in topics.head().iterrows():
            result += f"Topic {row['Topic']}: {row['Name']}\n"
        return result
    return "❌ No topics found."

# 🖼 Gradio Interface
def analyze_text(input_text, url=None, pdf_file=None, docx_file=None):
    # Main function that performs all analysis steps
    text = input_text

    if url:
        text = scrape_website(url)
    elif pdf_file:
        text = extract_text_from_pdf(pdf_file.name)
    elif docx_file:
        text = extract_text_from_docx(docx_file.name)

    if not text:
        return "❌ Could not extract text."

    label, confidence = detect_fake_news(text)
    sentiment = get_sentiment(text)
    sentiment_display = display_sentiment(sentiment)

    emotions = get_emotions(text)
    emotions_display = display_emotions(emotions)

    topics = extract_topics(text)
    topics_display = display_topics(topics)

    summary = summarize_text(text)
    summary_sentiment = get_sentiment(summary)
    summary_sentiment_display = display_sentiment(summary_sentiment)

    return {
        "Fake News Detection": f"📰 Fake News: {label} (Confidence: {confidence:.2f})",
        "Sentiment Analysis": sentiment_display,
        "Emotion Detection": emotions_display,
        "Topic Modeling": topics_display,
        "Summary": summary,
        "Summary Sentiment": summary_sentiment_display
    }

# 🖼 Gradio UI
iface = gr.Interface(
    fn=analyze_text,
    inputs=[
        gr.Textbox(label="Enter Text Here", placeholder="Type or paste your text for analysis", lines=4),
        gr.Textbox(label="Website URL (Optional)", placeholder="Enter website URL for analysis", lines=1),
        gr.File(label="Upload PDF (Optional)"),
        gr.File(label="Upload Word Document (Optional)")
    ],
    outputs=[
        gr.JSON(label="Analysis Results"),
    ],
    live=True,
    title="Universal Sentiment Analyzer",
    description="A powerful tool to analyze sentiment, emotion, fake news, topics, and summaries of various text sources.",
    theme="compact"
)

# Run the Gradio interface
if __name__ == "__main__":
    iface.launch()
