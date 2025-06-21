import logging
from scrape_headlines import scrape_headlines
from update_dataset import update_dataset
from clickbait_detector import train_model
from fetch_trending import fetch_trending_headlines
from eda import run_eda
import subprocess
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main workflow to run the clickbait detector project."""
    try:
        # Step 1: Scrape headlines
        logger.info("Starting headline scraping...")
        urls = [
            "https://www.buzzfeed.com",
            "https://www.bbc.com/news",
            "https://www.theguardian.com"
        ]
        for url in urls:
            df = scrape_headlines(url, max_headlines=10, use_selenium=True)
            if not df.empty:
                logger.info(f"Successfully scraped {len(df)} headlines from {url}")
                break
        else:
            logger.warning("No headlines scraped from any source")

        # Step 2: Update dataset
        logger.info("Updating dataset...")
        if not update_dataset():
            logger.error("Dataset update failed")
            return

        # Step 3: Train model
        logger.info("Training model...")
        if not train_model():
            logger.error("Model training failed")
            return

        # Step 4: Fetch trending headlines
        logger.info("Fetching trending headlines...")
        api_key = "your_api_key"  # Replace with your News API key
        trending_df = fetch_trending_headlines(api_key)
        if trending_df.empty:
            logger.warning("No trending headlines fetched")

        # Step 5: Run EDA
        logger.info("Running EDA...")
        if not run_eda():
            logger.error("EDA failed")

        # Step 6: Launch Streamlit app
        logger.info("Launching Streamlit app...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

    except Exception as e:
        logger.error(f"Error in main workflow: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()