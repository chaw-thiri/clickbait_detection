import requests
import pandas as pd
import logging
from clickbait_detector import predict_clickbait

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_trending_headlines(api_key=None, query="news", max_results=10, output_file="trending_headlines.csv"):
    """
    Fetch trending headlines from News API and predict clickbait labels.
    
    Args:
        api_key (str): News API key (optional for fallback to local file)
        query (str): Search query for headlines
        max_results (int): Maximum number of headlines to fetch
        output_file (str): Path to save trending headlines
    """
    try:
        headlines = []
        if api_key:
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            
            for article in articles[:max_results]:
                headline = article.get("title", "")
                if headline:
                    label, confidence = predict_clickbait(headline)
                    if label:
                        headlines.append({"headline": headline, "label": label, "confidence": confidence})
                    else:
                        headlines.append({"headline": headline, "label": "Unknown", "confidence": 0.0})
        else:
            logger.warning("No API key provided, attempting to load from local file")
            try:
                df = pd.read_csv(output_file)
                headlines = df.to_dict("records")
                logger.info(f"Loaded {len(headlines)} headlines from {output_file}")
            except FileNotFoundError:
                logger.error(f"{output_file} not found")
                return pd.DataFrame()
        
        df = pd.DataFrame(headlines)
        if not df.empty:
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} headlines to {output_file}")
        else:
            logger.warning("No headlines fetched")
        
        return df
    except Exception as e:
        logger.error(f"Error fetching or labeling headlines: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    api_key = "same_api_key_as_in_main.py"  # Replace with your News API key
    trending_df = fetch_trending_headlines(api_key)
    if not trending_df.empty:
        print(trending_df[["headline", "label"]])
    else:
        print("No headlines fetched. Check API key or model availability.")