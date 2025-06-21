import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scrape_headlines(url, max_headlines=10, use_selenium=True, output_file="scraped_headlines.csv"):
    """
    Scrape headlines from a given URL and label them as Clickbait or Non-Clickbait.
    
    Args:
        url (str): Website URL to scrape
        max_headlines (int): Maximum number of headlines to collect
        use_selenium (bool): Use Selenium for dynamic content
        output_file (str): Path to save scraped headlines
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.7151.120 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/"
    }
    headlines = []

    try:
        if use_selenium:
            logger.info(f"Using Selenium to fetch {url}")
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),  # Auto-detect Chrome version
                options=options
            )
            driver.get(url)
            time.sleep(5)  # Wait for JavaScript
            soup = BeautifulSoup(driver.page_source, "html.parser")
            driver.quit()
        else:
            logger.info(f"Fetching {url} with requests")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

        # Save debug HTML
        debug_filename = f"debug_{url.replace('https://', '').replace('/', '_')}.html"
        with open(debug_filename, "w", encoding="utf-8") as f:
            f.write(soup.prettify())
        logger.debug(f"Saved debug HTML to {debug_filename}")

        # Site-specific selectors
        site_selectors = {
            "reuters.com": (["a", "h1", "h2", "h3", "h4"], re.compile("text|media|headline|story|article|promo|card", re.I)),
            "bbc.com": (["a", "h2", "h3"], re.compile("promo|gs-c-promo|sc-4e3a3e86|sc-2e6dc1a3|headline|title", re.I)),
            "buzzfeed.com": (["a", "h2", "h3"], re.compile("buzz|list|quiz|article|title", re.I)),
            "theguardian.com": (["a", "h2", "h3"], re.compile("headline|title|article|story", re.I))
        }
        
        domain = next((d for d in site_selectors if d in url.lower()), None)
        selectors, class_pattern = site_selectors.get(domain, (["a", "h1", "h2", "h3", "h4"], re.compile("head|title|link|article|story", re.I)))

        # Scrape headlines
        for tag in soup.find_all(selectors, class_=class_pattern):
            text = tag.get_text().strip()
            if text and len(text) > 10 and text not in [h["headline"] for h in headlines]:
                clickbait_keywords = ["!", "shock", "unbelievable", "you won't believe", "secret", "amazing", "trick"]
                label = "Clickbait" if any(keyword in text.lower() for keyword in clickbait_keywords) else "Non-Clickbait"
                headlines.append({"headline": text, "label": label})
                logger.info(f"Found headline: {text} -> {label}")
            if len(headlines) >= max_headlines:
                break

        # Fallback selector
        if not headlines:
            for tag in soup.find_all(["a", "h1", "h2", "h3"], string=re.compile(".{10,}", re.I)):
                text = tag.get_text().strip()
                if text and len(text) > 10 and text not in [h["headline"] for h in headlines]:
                    label = "Clickbait" if any(keyword in text.lower() for keyword in clickbait_keywords) else "Non-Clickbait"
                    headlines.append({"headline": text, "label": label})
                    logger.info(f"Found fallback headline: {text} -> {label}")
                if len(headlines) >= max_headlines:
                    break

        df = pd.DataFrame(headlines)
        if not df.empty:
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} headlines to {output_file}")
        else:
            logger.warning(f"No headlines found at {url}. Inspect {debug_filename} or try another site.")
        
        return df

    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    urls = [
        "https://www.buzzfeed.com",
        "https://www.bbc.com/news",
        "https://www.theguardian.com"
    ]
    for url in urls:
        df = scrape_headlines(url, max_headlines=10, use_selenium=True)
        if not df.empty:
            break
    print(f"Scraped {len(df)} headlines")