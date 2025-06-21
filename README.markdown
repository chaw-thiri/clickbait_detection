# Clickbait Detector: An AI-Powered Headline Classification System


Welcome to the **Clickbait Detector**, a sophisticated machine learning project designed to classify news headlines as "Clickbait" or "Non-Clickbait" using advanced natural language processing (NLP) techniques. This project showcases the seamless integration of multiple technologies—web scraping, transformer-based NLP models, real-time data fetching, data visualization, and interactive web applications—to deliver a robust and user-friendly solution. Developed as a portfolio piece to demonstrate expertise in AI, data science, and full-stack development, this project is ideal for clients seeking innovative, data-driven solutions.

## Project Overview

The Clickbait Detector leverages a fine-tuned DistilBERT model to analyze headlines with high accuracy, distinguishing sensationalist clickbait from legitimate news. The system integrates a pipeline of tools for data collection, model training, real-time analysis, and user interaction, making it a comprehensive showcase of modern AI and web technologies. Key features include:

- **Web Scraping**: Dynamically extracts headlines from news websites using Selenium and BeautifulSoup.
- **Real-Time Data Fetching**: Retrieves trending headlines via the News API for up-to-date analysis.
- **NLP Model**: Employs a DistilBERT transformer model for precise headline classification.
- **Interactive Web App**: Built with Streamlit, allowing users to predict headlines, correct labels, and visualize insights.
- **Data Visualization**: Generates word clouds and label distribution charts using Plotly and WordCloud.
- **Dataset Management**: Combines scraped, corrected, and original data into a unified dataset with deduplication.

## Features

- **Headline Classification**: Predicts whether a headline is clickbait or non-clickbait with confidence scores using a fine-tuned DistilBERT model.
- **Dynamic Web Scraping**: Collects headlines from sites like BBC, Reuters, and BuzzFeed, with robust error handling and site-specific selectors.
- **Trending News Analysis**: Fetches and classifies trending headlines in real-time using the News API.
- **User Interaction**: Allows users to input custom headlines, correct model predictions, and save corrections via a Streamlit app.
- **Exploratory Data Analysis (EDA)**: Visualizes dataset insights with interactive bar charts and word clouds.
- **Automated Workflow**: Orchestrates scraping, dataset updates, model training, and app launch through a single script.
- **Robust Error Handling**: Includes comprehensive logging to ensure reliability across components.

## Technology Stack

The Clickbait Detector integrates a diverse set of technologies, demonstrating proficiency in AI, data processing, and web development:

- **Machine Learning & NLP**:
  - **PyTorch**: Powers the DistilBERT model for training and inference.
  - **Transformers (Hugging Face)**: Provides pre-trained DistilBERT models and tokenizers for NLP tasks.
  - **scikit-learn**: Used for dataset splitting and performance metrics (accuracy, F1-score).

- **Web Scraping**:
  - **Selenium**: Handles dynamic content rendering for modern websites.
  - **BeautifulSoup**: Parses HTML to extract headlines with site-specific selectors.
  - **requests**: Fetches static web content efficiently.

- **Data Processing**:
  - **pandas**: Manages datasets, including merging, deduplication, and filtering.
  - **NumPy**: Supports numerical operations during data preprocessing.

- **Web Development**:
  - **Streamlit**: Builds an interactive web app for user input, predictions, and visualizations.
  - **Plotly**: Creates interactive bar charts for label distribution.
  - **WordCloud**: Generates visual representations of headline content.

- **API Integration**:
  - **News API**: Retrieves real-time trending headlines for analysis.

- **System Integration**:
  - **logging**: Implements comprehensive logging for debugging and monitoring.
  - **subprocess**: Orchestrates script execution in the main workflow.
  - **webdriver-manager**: Automates ChromeDriver setup for Selenium.

- **Development Tools**:
  - **Python**: Core programming language for all components.
  - **Git**: Version control for project management.

This technology stack highlights the project’s ability to integrate front-end, back-end, and AI components into a cohesive system, making it a standout example for Upwork clients.

## Project Structure

The project is organized into modular scripts, each handling a specific component of the pipeline:

- **`main.py`**: Orchestrates the entire workflow, from scraping to launching the Streamlit app.
- **`scrape_headlines.py`**: Scrapes headlines from news websites using Selenium and BeautifulSoup.
- **`update_dataset.py`**: Combines scraped and corrected headlines into the main dataset, ensuring no duplicates.
- **`clickbait_detector.py`**: Trains and deploys the DistilBERT model for headline classification.
- **`fetch_trending.py`**: Fetches trending headlines via News API and predicts their labels.
- **`eda.py`**: Generates word clouds and label distribution charts for dataset insights.
- **`app.py`**: Runs the Streamlit app for user interaction and visualization.

**Data Files**:
- `headlines_dataset.csv`: Main dataset containing headlines and labels.
- `scraped_headlines.csv`: Temporary storage for scraped headlines.
- `corrected_headlines.csv`: Stores user-corrected labels from the Streamlit app.
- `trending_headlines.csv`: Contains fetched trending headlines with predicted labels.
- `clickbait_wordcloud.png` & `non_clickbait_wordcloud.png`: Visualizations of headline content.

## Installation

To run the Clickbait Detector locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chaw-thiri/clickbait-detector.git
   cd clickbait-detector
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed, then install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install:
   ```bash
   pip install streamlit pandas torch transformers requests beautifulsoup4 selenium webdriver_manager wordcloud matplotlib plotly
   ```

3. **Prepare the Dataset**:
   Ensure `headlines_dataset.csv` exists in the project root with at least some initial data (format: `headline,label`).

4. **Obtain a News API Key**:
   - Sign up at [News API](https://newsapi.org/) to get an API key.
   - Update `main.py` and `fetch_trending.py` with your API key, or leave it blank to use local fallback.

5. **Run the Project**:
   Execute the main workflow script:
   ```bash
   python main.py
   ```
   This will:
   - Scrape headlines from news websites.
   - Update the dataset.
   - Train the model (if needed).
   - Fetch trending headlines.
   - Generate EDA visuals.
   - Launch the Streamlit app (accessible at `http://localhost:8501`).

## Usage

### Streamlit App
The Streamlit app provides an intuitive interface for interacting with the Clickbait Detector:
- **Predict a Headline**: Enter a headline to get a clickbait/non-clickbait prediction with a confidence score.
- **Fetch Trending Headlines**: Input a News API key to retrieve and classify trending news, or use the local `trending_headlines.csv`.
- **Correct Predictions**: Adjust model predictions and save corrections to `corrected_headlines.csv`.
- **View Dataset Insights**: Explore label distribution charts and word clouds for clickbait and non-clickbait headlines.

### Command-Line Usage
Individual scripts can be run for specific tasks:
- Scrape headlines: `python scrape_headlines.py`
- Update dataset: `python update_dataset.py`
- Train model: `python clickbait_detector.py`
- Fetch trending headlines: `python fetch_trending.py`
- Generate EDA visuals: `python eda.py`

## Integration Highlights

The Clickbait Detector stands out due to its seamless integration of diverse technologies:

- **Web Scraping + NLP**: Combines Selenium/BeautifulSoup for data collection with DistilBERT for classification, enabling end-to-end headline analysis.
- **Real-Time Data + AI**: Integrates News API for live data with a pre-trained model for instant predictions.
- **Frontend + Backend**: Streamlit provides a user-friendly interface, while pandas and PyTorch handle data processing and model training in the backend.
- **Visualization + Insights**: Plotly and WordCloud deliver interactive and visual insights, enhancing data interpretability.
- **Modular Design**: Each component (scraping, training, app) is modular, allowing easy updates or extensions.

This integration showcases the ability to build complex, full-stack AI solutions, a valuable skill for Upwork clients in industries like media, marketing, and data science.




