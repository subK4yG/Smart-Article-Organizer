import streamlit as st
import random
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import SGDClassifier
from fuzzywuzzy import fuzz
import plotly.express as px
# Page configuration
st.set_page_config(layout="wide")
ENGINEERING_SUBJECTS = {
    "Quantum Computing & Information": {
        "keywords": ["qubit", "quantum computing", "quantum processor", "quantum circuit", 
                    "superconducting", "quantum gate", "quantum algorithm", "entanglement", "decoherence", "quantum information", "quantum communication", "quantum cryptography",
                    "quantum key distribution", "quantum network", "quantum teleportation"],
        "icon": "💻"
    },
    "Quantum Machine Learning": {
        "keywords": ["machine learning", "AI", "neural network", "deep learning", 
                    "quantum neural network", "quantum AI", "reinforcement learning"],
        "icon": "🧠"
    },
    "Quantum Hardware & Circuits": {
        "keywords": ["quantum hardware", "quantum device", "quantum chip", 
                    "quantum resonator", "quantum measurement", "qubit fabrication"],
        "icon": "⚙️"
    },
    "Mathematical & Theoretical Quantum Physics": {
        "keywords": ["mathematical", "theory", "quantum field", "quantum mechanics",
                    "quantum model", "quantum equation", "wavefunction"],
        "icon": "📐"
    },
    "Quantum Cryptography & Security": {
        "keywords": ["cryptography", "security", "encryption", "quantum-safe",
                    "post-quantum", "quantum-resistant", "secure communication"],
        "icon": "🔒"
    },
    "Quantum Simulation": {
        "keywords": ["simulation", "quantum simulator", "quantum dynamics",
                    "quantum system", "quantum model", "quantum annealing"],
        "icon": "🔄"
    }
}
subject_icons = {subject: data["icon"] for subject, data in ENGINEERING_SUBJECTS.items()}
# Normalize Text
def normalize(text):
    return text.strip().lower()
# Load Data
file_path = "engg_articles.csv"
try:
    df = pd.read_csv(file_path)
    if df.empty:
        st.error("The dataset is empty. Please check the file content.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
# Remove Predicted Subject Column if it exists
if "Predicted Subject" in df.columns:
    df.drop(columns=["Predicted Subject"], inplace=True)
# Handle NLP Model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error(f"Error loading SpaCy model: {e}")
    st.stop()
# Match Subject using Fuzzy Matching
def match_subject(text):
    text = normalize(text)
    for subject, data in ENGINEERING_SUBJECTS.items():
        for keyword in data['keywords']:
            if fuzz.token_set_ratio(text, normalize(keyword)) > 80:
                return subject, subject_icons[subject]
    return "Unknown", "❓"
# Classify Subject (Rule-based and ML Hybrid)
def classify_subject(row):
    text = f"{row['keywords']} {row['abstract']}".lower()
    # First attempt exact matching
    for subject, data in ENGINEERING_SUBJECTS.items():
        if any(keyword in text for keyword in data["keywords"]):
            return subject
    # Fallback to ML classification if vectorizer and classifier are available
    if 'vectorizer' in st.session_state and 'classifier' in st.session_state:
        text_vec = st.session_state.vectorizer.transform([text])
        return st.session_state.classifier.predict(text_vec)[0]
    return "Uncategorized"
# Apply initial classification
df["Predicted Subject"] = df.apply(classify_subject, axis=1)
# Perform Machine Learning Classification if classes are valid
unique_classes = df["Predicted Subject"].nunique()
if unique_classes > 1:
    # Remove classes with fewer than 2 samples
    class_counts = df["Predicted Subject"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df_filtered = df[df["Predicted Subject"].isin(valid_classes)]
    if len(valid_classes) < 2:
        st.warning("⚠️ Not enough valid classes for stratified train-test split.")
    else:
        X = df_filtered["keywords"].fillna("") + " " + df_filtered["abstract"].fillna("")
        y = df_filtered["Predicted Subject"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        # Vectorize Text Data
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        # Train Classifier
        classifier = SGDClassifier(loss='log_loss', random_state=42)
        classifier.fit(X_train_vec, y_train)
        # Store model and vectorizer
        st.session_state.vectorizer = vectorizer
        st.session_state.classifier = classifier
else:
    st.warning("⚠️ Insufficient categories for machine learning classification. Falling back to rule-based classification.")
    df["Predicted Subject"] = df.apply(classify_subject, axis=1)
def analyze_sentiment(text):
    # Check if text is missing or invalid
    if not text or not isinstance(text, str) or not text.strip():
        return "No Abstract"    
    # Calculate sentiment using TextBlob
    sentiment_score = TextBlob(text.strip()).sentiment.polarity
    # Return appropriate sentiment
    if sentiment_score > 0:
        return "😊 Positive"
    elif sentiment_score < 0:
        return "😟 Negative"
    else:
        return "😐 Neutral"
def sentiment_analysis():
    st.subheader("📊 Sentiment Analysis of Research Abstracts")
    # Check for valid abstracts
    if "abstract" not in df.columns:
        st.error("Column 'abstract' not found in the dataset.")
        return
    if df["abstract"].isnull().all():
        st.error("No abstracts available for sentiment analysis.")
        return
    # Perform sentiment analysis
    df["Sentiment"] = df["abstract"].apply(analyze_sentiment)
    sentiment_counts = df["Sentiment"].value_counts()
    # Plot results
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm", ax=ax)
    # Add data labels to bars
    for i, count in enumerate(sentiment_counts.values):
        ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=12)
    ax.set_xlabel("Sentiment", fontsize=14)
    ax.set_ylabel("Number of Articles", fontsize=14)
    ax.set_title("Sentiment Distribution of Research Abstracts", fontsize=16)
    st.pyplot(fig)
def most_common_keywords():
    st.subheader("🔑 Most Common Keywords")
    # Extract keywords safely
    if "keywords" not in df.columns:
        st.error("Keywords column not found.")
        return
    all_keywords = df["keywords"].dropna().str.lower().str.split(r'[;,.\s]+').explode()
    keyword_counts = Counter(all_keywords)    
    # Display most common keywords
    most_common_df = pd.DataFrame(keyword_counts.most_common(10), columns=["Keyword", "Count"])
    st.dataframe(most_common_df)
    # Plot visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Count", y="Keyword", data=most_common_df, palette="viridis", ax=ax)
    ax.set_title("Top 10 Most Common Keywords")
    st.pyplot(fig)
def most_cited_articles():
    st.subheader("🏆 Most Cited Research Papers")
    # Validate columns
    if "citations" not in df.columns or "title" not in df.columns or "authors" not in df.columns or "publication" not in df.columns:
        st.error("Required columns for citation analysis are missing.")
        return
    try:
        # Sort and display the top cited articles
        top_cited = df.sort_values(by="citations", ascending=False).head(10)
        st.dataframe(top_cited[["title", "authors", "citations", "publication"]])
    except Exception as e:
        st.error(f"Error displaying most cited articles: {e}")
# Custom CSS for Styling and Animation
st.markdown(
    """
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #1f1c2c, #928DAB);
            color: white;
        }
        .main-title {
            color: #0096FF;
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            margin-top: 20px;
            animation: glow 1.5s infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px #0096FF; }
            to { text-shadow: 0 0 20px #0056b3; }
        }
        .page-container {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.6);
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        }
        .page-container:hover {
            transform: scale(1.03);
            box-shadow: 0px 0px 40px rgba(255, 255, 255, 0.8);
        }
        .floating-box-home {
            background: linear-gradient(135deg, #ff7eb3, #ff758c);
            padding: 15px;
            border-radius: 10px;
            color: white;
            font-size: 18px;
            text-align: center;
            animation: float 3s infinite alternate ease-in-out;
        }
        .floating-box-subjects {
            background: linear-gradient(135deg, #0096FF, #0056b3);
            padding: 15px;
            border-radius: 10px;
            color: yellow;
            font-size: 18px;
            text-align: center;
            animation: float 3s infinite alternate ease-in-out;
        }
        .floating-box-years {
            background: linear-gradient(135deg, #32CD32, #228B22);
            padding: 15px;
            border-radius: 10px;
            color: white;
            font-size: 18px;
            text-align: center;
            animation: float 3s infinite alternate ease-in-out;
        }
        @keyframes float {
            from { transform: translateY(0px); }
            to { transform: translateY(-10px); }
        }
        .ripple-button {
            background-color: #0056b3;
            border: none;
            color: white;
            padding: 12px 24px;
            text-align: center;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        .ripple-button:active::after {
            content: '';
            background: rgba(255, 255, 255, 0.5);
            position: absolute;
            border-radius: 50%;
            transform: scale(0);
            animation: ripple 0.6s linear;
        }
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# ------------------- Home Page -------------------
def home_page():
    st.markdown("<div class='page-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>🏠 Welcome to Article Organizer! 📚📝</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class='floating-box-home'>✨ Organize, Search, and Explore Research Articles with Ease! 🚀</div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
# ------------------- Search Page -------------------
def search_page():
    st.markdown("<div class='page-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>🔎 Search Articles 🧐</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class='floating-box-subjects'>📂 Select Subjects to Filter</div>
        <div class='floating-box-years'>📅 Choose Publication Years</div>
    """, unsafe_allow_html=True)
    # Ensure publication is in date format
    df['publication'] = pd.to_datetime(df['publication'], errors='coerce').dt.year
    # Select search filters
    selected_subjects = st.multiselect("📂 Select Subjects:", df["category"].dropna().unique())
    selected_years = st.multiselect("📅 Select Years:", sorted(df["publication"].dropna().unique().astype(int)))
    search_query = st.text_input("🔍 Search by Keywords:")
    # Filter Data
    filtered_df = df.copy()
    if selected_subjects:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_subjects)]
    if selected_years:
        filtered_df = filtered_df[filtered_df["publication"].isin(selected_years)]
    if search_query:
        filtered_df = filtered_df[filtered_df["keywords"].fillna('').str.contains(search_query, case=False, na=False)]
    # Display Results
    if not filtered_df.empty:
        st.dataframe(filtered_df.drop(columns=["Predicted Subject"], errors='ignore'), use_container_width=True)
    else:
        st.warning("⚠️ No results found. Please adjust your search filters.")  
    st.markdown("</div>", unsafe_allow_html=True)
# ------------------- Delete Article Page -------------------
def delete_article_page():
    st.markdown("<div class='page-container'>", unsafe_allow_html=True)
    st.title("🗑️ Delete an Article")
    global df    
    if df.empty:
        st.warning("⚠️ No articles found in the dataset.")
        return
    selected_article = st.selectbox("Select an article to delete:", df["title"].unique())
    if st.button("🗑️ Delete Article"):
        subject_category = df[df["title"] == selected_article]["category"].values[0]
        remaining_articles = df[df["category"] == subject_category]
        if len(remaining_articles) <= 2:
            st.warning(f"⚠️ Cannot delete this article! At least 2 articles are required in '{subject_category}' for classification.")
        else:
            df = df[df["title"] != selected_article]
            df.to_csv(file_path, index=False)
            st.success(f"✅ Article '{selected_article}' deleted successfully!")            
    st.markdown("</div>", unsafe_allow_html=True)
# ------------------- Create New Subject -------------------
def create_new_subject(subject_name, keywords):
    """Creates a new subject folder with given name and keywords."""
    if subject_name not in ENGINEERING_SUBJECTS:
        ENGINEERING_SUBJECTS[subject_name] = {
            "keywords": [kw.strip().lower() for kw in keywords.split(",")],
            "icon": "📁"  # Default icon
        }
        subject_icons[subject_name] = "📁"
        st.success(f"✅ New subject folder '{subject_name}' created successfully!")
        return True
    else:
        st.warning(f"⚠️ Subject folder '{subject_name}' already exists!")
        return False
# ------------------- Subject Folders Page -------------------
def subject_folders_page():
    st.markdown("<div class='page-container'>", unsafe_allow_html=True)
    st.title("📂 Subject Folders 📚")
    # Section to create a new subject folder
    with st.expander("➕ Create New Subject Folder"):
        new_subject = st.text_input("New Subject Name:")
        new_keywords = st.text_input("Associated Keywords (comma-separated):")
        if st.button("Create Folder"):
            if new_subject and new_keywords:
                if create_new_subject(new_subject, new_keywords):
                    # Reclassify articles with new subject
                    df['category'] = df.apply(lambda row: classify_subject(row['keywords'] + " " + row['abstract']), axis=1)
                    st.success(f"✅ Folder '{new_subject}' created and articles reclassified.")
            else:
                st.warning("⚠️ Please enter both subject name and keywords!")
    # Ensure categories are stored as lists
    df['category'] = df['category'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)
    # Extract unique subjects
    all_subjects = sorted({subject for categories in df['category'] for subject in categories})
    # Display subject folders
    for subject in all_subjects:
        icon = subject_icons.get(subject, "📖")
        st.markdown(f"<div class='floating-box-subjects'><h2>{icon} {subject}</h2></div>", unsafe_allow_html=True)
        # Filter articles belonging to this subject
        subject_articles = df[df['category'].apply(lambda x: subject in x)]
        if not subject_articles.empty:
            st.write(f"**Articles in {subject}:** {len(subject_articles)}")
            st.dataframe(subject_articles[['title', 'authors', 'publication', 'citations']], use_container_width=True)
        else:
            st.warning(f"No articles found under **{subject}**")
    st.markdown("</div>", unsafe_allow_html=True)
# ------------------- Year Folders Page -------------------
def year_folders_page():
    st.markdown("<div class='page-container'>", unsafe_allow_html=True)
    st.title("📅 Year Folders 📂")
    # Display folders by publication years
    years = sorted(df["publication"].unique(), reverse=True)
    for year in years:
        year_articles = df[df["publication"] == year]
        article_count = len(year_articles)
        # Show year folder with article count
        st.markdown(f"""
            <div class='floating-box-years'>
                <h2>📆 {year}</h2>
                <p>📄 Articles: {article_count}</p>
            </div>
        """, unsafe_allow_html=True)
        # Display articles if available
        if article_count > 0:
            st.dataframe(year_articles[['title', 'authors', 'category', 'citations']], use_container_width=True)
        else:
            st.warning(f"No articles found for the year **{year}**")
    st.markdown("</div>", unsafe_allow_html=True)
# ------------------- Add Article Page -------------------
def add_article_page():
    global df 
    st.markdown("<div class='page-container'>", unsafe_allow_html=True)
    st.title("📝 Add a New Article ✍️")
    # Input Fields
    title = st.text_input("📌 Article Title")
    abstract = st.text_area("📖 Abstract")
    authors = st.text_input("✍️ Authors (comma-separated)")
    publication_year = st.number_input("📅 Publication Year", min_value=1900, max_value=2025, value=2023)
    citations = st.number_input("📊 Citations Count", min_value=0, value=0)
    keywords = st.text_area("🔍 Keywords (comma-separated)")
    url = st.text_input("🔗 Article URL")    
    # Category Options
    existing_categories = df['category'].dropna().unique().tolist()
    existing_categories.sort()
    category_input = st.selectbox("📚 Select Existing Category or Enter a New One:", ["Create New Category"] + existing_categories)
    # Allow Manual Category Entry
    if category_input == "Create New Category":
        new_category = st.text_input("🆕 Enter New Category Name:")
        if new_category:
            category_input = new_category
    # Add Article Button
    if st.button("➕ Add Article"):       
        # Final category assignment (manual or auto-classified)
        if not category_input:
            subject_category = classify_subject(f"{keywords} {abstract}")
        else:
            subject_category = category_input
        new_article = pd.DataFrame([[
            title, authors, abstract, url, keywords, subject_category, citations, publication_year
        ]], columns=["title", "authors", "abstract", "url", "keywords", "category", "citations", "publication"])
        # Append to the dataframe
        df = pd.concat([df, new_article], ignore_index=True)
        df.to_csv(file_path, index=False)
        st.success(f"✅ Article '{title}' added successfully under category '{subject_category}'!")
    st.markdown("</div>", unsafe_allow_html=True)
def data_analysis_page():
    st.markdown("<div class='page-container'>", unsafe_allow_html=True)
    st.title("📊 Data Analysis and Visualization")
    # ------------------- Subject Distribution  -------------------
    st.subheader("📌Distribution of Articles by Subject")
    # Create a set for accurate count by avoiding duplication
    def clean_categories(categories):
        return categories.split(', ') if isinstance(categories, str) else []
    all_subjects = df['category'].dropna().apply(clean_categories)
    unique_subjects = all_subjects.explode().value_counts()
    fig_unique = px.pie(
        values=unique_subjects.values,
        names=unique_subjects.index,
        title="Subject Distribution",
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_unique)
    # ------------------- Multi-Category Distribution -------------------
    st.subheader("🔎 Articles Belonging to Multiple Categories")
    df['category_count'] = df['category'].str.count(', ') + 1
    multi_category_distribution = df['category_count'].value_counts().sort_index()
    fig_multi = px.bar(
        x=multi_category_distribution.index,
        y=multi_category_distribution.values,
        labels={'x': 'Number of Categories', 'y': 'Number of Articles'},
        title="Number of Articles with Multiple Categories",
        color=multi_category_distribution.index,
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_multi)
    # Year-wise Distribution of Articles
    st.subheader("📅 Year-wise Distribution of Articles")
    year_counts = df['publication'].value_counts().sort_index()
    # Plot the Year-wise Bar Chart
    fig = px.bar(
        x=year_counts.index,
        y=year_counts.values,
        labels={"x": "Publication Year", "y": "Number of Articles"},
        title="📅 Year-wise Distribution of Articles",
        color=year_counts.values,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig)
    # Word Cloud for Keywords
    st.subheader("☁️ Word Cloud for Keywords")
    if df["keywords"].notna().sum() > 0:
        all_keywords = ' '.join(df["keywords"].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("⚠️ No keywords available for generating a word cloud.")
    # Top Contributing Authors
    st.subheader("📚 Top Contributing Authors")
    if df["authors"].notna().sum() > 0:
        all_authors = df["authors"].dropna().str.split(",").explode().str.strip()
        author_counts = all_authors.value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(y=author_counts.index, x=author_counts.values, palette="viridis", ax=ax)
        ax.set_xlabel("Number of Publications")
        ax.set_ylabel("Authors")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No author data available for analysis.")
    st.markdown("</div>", unsafe_allow_html=True)
# ✅ Download Articles
def download_articles():
    st.subheader("📥 Download Selected Articles")
    selected_articles = st.multiselect("Select articles to download:", df["title"])       
    if selected_articles:
        filtered_df = df[df["title"].isin(selected_articles)]       
        # Remove "Predicted Subject" column if it exists
        if "Predicted Subject" in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=["Predicted Subject"])        
        # Convert to CSV and download
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Selected Articles 📥", csv, "selected_articles.csv", "text/csv")
    else:
        st.info("Please select at least one article to download.")
# ✅ Summarize Articles
nlp = spacy.load("en_core_web_sm")
def summarize_article(text):
    doc = nlp(text)
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            word_frequencies[word.text] = word_frequencies.get(word.text, 0) + 1
    if not word_frequencies:
        return "Not enough content to generate a summary."
    max_freq = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_freq
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text]
    summarized_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)
    summary = " ".join([sent.text for sent in summarized_sentences])
    return summary
def show_summary():
    st.subheader("🔍 AI-Powered Article Summary")
    article_text = st.text_area("Paste the research paper abstract:")
    if st.button("Summarize"):
        summary = summarize_article(article_text)
        st.write(summary)
# ✅ Train Citation Model
def train_citation_model():
    df_clean = df.dropna(subset=["publication", "citations"])
    X = df_clean[["publication"]].values
    y = df_clean["citations"].values
    model = LinearRegression()
    model.fit(X, y)
    return model
citation_model = train_citation_model()
# ✅ Predict Citations
def predict_citations():
    st.subheader("📊 Predict Future Citations")
    year = st.slider("Select a future year:", min_value=2025, max_value=2035, value=2030)
    if st.button("🔍 Predict Citations"):
        predicted_citations = citation_model.predict(np.array([[year]]))[0]
        st.success(f"📌 Predicted Citations by {year}: {int(predicted_citations)}")
# ✅ Cluster Research Topics
vectorizer = TfidfVectorizer(stop_words="english")
article_vectors = vectorizer.fit_transform(df["keywords"].dropna())
def cluster_research_topics():
    st.subheader("🔍 Research Topic Clustering")
    num_clusters = st.slider("Select Number of Clusters:", 2, 10, 5)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df.loc[df["keywords"].notna(), "Cluster"] = kmeans.fit_predict(article_vectors)
    st.write("\n🧠 **Clustered Research Topics:**")
    for cluster in range(num_clusters):
        cluster_papers = df[df["Cluster"] == cluster]["title"].head(5).tolist()
        st.markdown(f"**Cluster {cluster + 1}:**")
        for paper in cluster_papers:
            st.write(f"- {paper}")
# ✅ Combined Page
def citation_and_clustering_page():
    st.title("📊 Citation Impact & Research Topic Clustering")
    predict_citations()
    cluster_research_topics()
st.sidebar.title("🔍 Navigation 🚀")
page = st.sidebar.selectbox("📌 Choose Page", ["🏠 Home 🏡", "🔎 Search Articles 🔍", "📂 Subject Folders 📁", "📅 Year Folders 📆", "📝 Add Article ✨", "🗑️ Delete Article", "📊 Data Analysis 📈", "📥 Download Articles",   "📊 Citation Prediction & Clustering 📑",  "🔍 AI Summary 📄"])
if page == "🏠 Home 🏡":
    home_page()
elif page == "🔎 Search Articles 🔍":
    search_page()
elif page == "📂 Subject Folders 📁":
    subject_folders_page()
elif page == "📅 Year Folders 📆":
    year_folders_page()
elif page == "📝 Add Article ✨":
    add_article_page()
elif page == "🗑️ Delete Article":
    delete_article_page()
elif page == "📊 Data Analysis 📈":
    data_analysis_page()
    sentiment_analysis()
    most_common_keywords()
    most_cited_articles()
elif page == "📥 Download Articles":
    download_articles()
elif page == "📊 Citation Prediction & Clustering 📑":
    citation_and_clustering_page()    
elif page == "🔍 AI Summary 📄":
    show_summary()
