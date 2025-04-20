# 🧠 Universal Sentiment Analyzer

Welcome to the **Universal Sentiment Analyzer** – a personal project that brings multiple natural language processing (NLP) features into one place. Whether you're analyzing your own writing, reviewing online articles, or just curious about the emotional tone of a document, this tool makes it easy and insightful.

It’s designed to be beginner-friendly, with clear results, easy input options, and multiple powerful models working in the background — all in one interface.

---

## ✨ What This Tool Can Do

### 🔹 1. Sentiment Analysis (VADER)
Want to quickly understand the tone of a sentence or paragraph? This feature tells you:
- How **positive**, **neutral**, or **negative** your text is.
- It also shows a **compound score** that gives an overall sentiment in one number.

### 🔹 2. Emotion Detection (Triple Engine)
Using **three different methods**, the tool extracts emotional tones:
- **Text2Emotion**: Recognizes Happy, Angry, Surprise, Sad, and Fear.
- **DistilRoBERTa**: A deep learning model trained on emotions to give you refined emotion labels.
- **NRCLex**: A rule-based method using emotion lexicons to score the intensity of emotions.

### 🔹 3. Fake News Detection
Curious whether a piece of news might be fake? Paste it into the tool and it will predict whether it’s likely **fake or real**, using a BERT-based model.

### 🔹 4. Text Summarization
Long articles? No problem. This tool uses the **BART transformer model** to generate a short summary that keeps the core meaning intact.

### 🔹 5. Topic Modeling (BERTopic)
Want to discover what your document is about? This tool finds key **topics** and **keywords** using BERTopic, giving you a quick sense of the subject.

---

## 📝 How You Can Provide Input
This tool accepts multiple types of input so you don’t have to limit yourself:
- 📄 **Type or paste text** manually.
- 🌍 **Paste a URL** to extract article content from a website.
- 📑 **Upload a PDF** file for text extraction and analysis.
- 📃 **Upload a Word (.docx)** file as well.

All inputs are handled automatically. Just pick the one that suits you.

---

## ⚙️ Getting Started (Setup Instructions)

### Step 1: Clone the Project
```bash
git clone https://github.com/your-username/universal-sentiment-analyzer.git
cd universal-sentiment-analyzer
```

### Step 2: Install Required Libraries
Use pip to install all dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

Here's what's inside the `requirements.txt`:
```txt
gradio
requests
beautifulsoup4
vaderSentiment
transformers
textblob
bertopic
text2emotion
nrclex
python-docx
PyPDF2
```

### Step 3: Launch the App
```bash
python app.py
```

This will open a Gradio web interface in your browser where you can start analyzing right away.

---

## 📊 How It Works Behind the Scenes

Here’s a simple breakdown of what happens when you enter text:
1. **Input Handling**: It checks whether you provided a URL, PDF, Word file, or just plain text.
2. **Sentiment Analysis**: It uses VADER to classify the tone.
3. **Emotion Detection**: Three models extract emotional tones from the same text.
4. **Fake News Prediction**: A BERT model checks for misinformation.
5. **Topic Extraction**: BERTopic finds relevant topics.
6. **Summary Generation**: The BART model summarizes the text.
7. **Output Display**: You get a clean, readable dashboard with all results.

---

## 🤖 Models Used
- **VADER** – Sentiment analysis
- **Text2Emotion** – Emotion detection
- **DistilRoBERTa** – Emotion classification
- **NRCLex** – Lexicon-based emotion scoring
- **BERT Tiny** – Fake news detection
- **BART Large CNN** – Summarization
- **BERTopic** – Topic modeling

All of these are loaded using the `transformers`, `bertopic`, or `nrclex` libraries.

---

## 📌 Notes
- Some models (like BART and BERTopic) might take a few seconds to process.
- Summarization is skipped for very short inputs.
- The first time you run the app, some models may take time to load.
- Requires internet connection for the first run (to download models).

---

## 📄 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🤝 Want to Contribute or Give Feedback?
Feel free to fork, submit issues, or suggest improvements. If you find it useful, a star ⭐️ would mean a lot!

---

Thanks for checking out my project. I created this to blend multiple NLP tasks into a user-friendly interface, and I hope it helps you explore and understand text in a deeper way!

– Manasi 😊

