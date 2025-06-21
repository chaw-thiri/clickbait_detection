import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_dataset(scraped_file="scraped_headlines.csv", corrections_file="corrected_headlines.csv", output_file="headlines_dataset.csv"):
    """
    Update the main dataset with scraped headlines and user corrections, removing duplicates.
    
    Args:
        scraped_file (str): Path to scraped headlines CSV
        corrections_file (str): Path to corrected headlines CSV
        output_file (str): Path to main dataset CSV
    """
    try:
        # Load main dataset
        original_df = pd.read_csv(output_file)
        logger.info(f"Loaded main dataset with {len(original_df)} samples")
        
        # Initialize list of DataFrames to combine
        dfs = [original_df]
        
        # Add scraped headlines if file exists
        try:
            scraped_df = pd.read_csv(scraped_file)
            if not scraped_df.empty:
                dfs.append(scraped_df)
                logger.info(f"Added {len(scraped_df)} scraped headlines")
        except FileNotFoundError:
            logger.warning(f"{scraped_file} not found, skipping")
        
        # Add corrections if file exists
        try:
            corrections_df = pd.read_csv(corrections_file)
            if not corrections_df.empty:
                dfs.append(corrections_df)
                logger.info(f"Added {len(corrections_df)} corrected headlines")
        except FileNotFoundError:
            logger.warning(f"{corrections_file} not found, skipping")
        
        # Combine and remove duplicates
        if len(dfs) > 1:
            combined_df = pd.concat(dfs).drop_duplicates(subset="headline").reset_index(drop=True)
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Updated dataset with {len(combined_df)} total samples")
            logger.info(f"Label distribution:\n{combined_df['label'].value_counts().to_string()}")
        else:
            logger.info("No new data to add, dataset unchanged")
        
        return True
    except FileNotFoundError:
        logger.error(f"Main dataset {output_file} not found")
        return False
    except Exception as e:
        logger.error(f"Error updating dataset: {e}")
        return False

if __name__ == "__main__":
    update_dataset()