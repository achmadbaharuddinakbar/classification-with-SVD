import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify, render_template

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    # 1. Case Folding: Mengubah semua huruf menjadi huruf kecil
    text = text.lower()
    
    # 2. Menghilangkan angka dan karakter khusus
    text = re.sub(r'\d+', '', text)  # Menghapus angka
    text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca dan karakter khusus
    
    # 3. Tokenisasi: Memecah teks menjadi kata-kata
    words = word_tokenize(text)
    
    # 4. Menghapus stopwords: kata-kata umum yang tidak membawa banyak informasi
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]
    
    # 5. Stemming: Mengubah kata ke bentuk dasar menggunakan Sastrawi
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Menggabungkan kata-kata kembali menjadi satu kalimat
    processed_text = ' '.join(words)
    
    return processed_text

# Memuat model dan pipeline yang telah dilatih
pipeline = joblib.load('svd_tfidf_pipeline.pkl')

# Inisialisasi Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    # Ambil input dari form
    user_input = request.form['text']
    
    if user_input:
        # Preprocessing input
        preprocessed_text = preprocess_text(user_input)
        
        # Melakukan prediksi
        prediction = pipeline.predict([preprocessed_text])
        predicted_category = "Kesehatan" if prediction[0] == 0 else "Kuliner"
        
        # Mengembalikan hasil prediksi
        return jsonify({'result': f'Hasil Klasifikasi: {predicted_category}'})
    else:
        return jsonify({'error': 'Silakan masukkan teks berita terlebih dahulu.'})

if __name__ == "__main__":
    app.run(debug=True)