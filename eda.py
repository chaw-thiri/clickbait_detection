import pandas as pd
try:
    from wordcloud import WordCloud
except ImportError:
    print("Error: 'wordcloud' module not found. Install it using 'pip install wordcloud'")
    exit(1)
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_wordcloud(df, label, filename):
    """
    Generate and save a word cloud for headlines of a specific label.
    
    Args:
        df (pd.DataFrame): Dataset with headlines and labels
        label (str): Label to filter headlines (e.g., "Clickbait")
        filename (str): Path to save the word cloud image
    """
    try:
        text = " ".join(df[df["label"] == label]["headline"].values)
        if not text:
            logger.warning(f"No headlines found for label: {label}")
            return False
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(filename)
        plt.close()
        logger.info(f"Generated word cloud for {label} at {filename}")
        return True
    except Exception as e:
        logger.error(f"Error generating wordcloud for {label}: {e}")
        return False

def plot_label_distribution(df):
    """
    Create a bar chart of label distribution.
    
    Args:
        df (pd.DataFrame): Dataset with labels
    
    Returns:
        plotly.graph_objects.Figure: Bar chart of label distribution
    """
    try:
        counts = df["label"].value_counts()
        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts.index.tolist(),
                    y=counts.values.tolist(),
                    marker=dict(color=["#FF6B6B", "#4ECDC4", "#FFD700"], line=dict(color=["#D9534F", "#3AB09E", "#E6B800"], width=1)),
                )
            ],
            layout=dict(
                title="Distribution of Headlines by Label",
                xaxis=dict(title="Label"),
                yaxis=dict(title="Number of Headlines", zeroline=True),
                showlegend=False
            )
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating label distribution: {e}")
        return None

def run_eda(dataset_file="headlines_dataset.csv"):
    """
    Run EDA to generate word clouds and label distribution.
    
    Args:
        dataset_file (str): Path to dataset CSV
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        df = pd.read_csv(dataset_file)
        if df.empty or "headline" not in df.columns or "label" not in df.columns:
            logger.error("Invalid or empty dataset")
            return False
        
        # Generate word clouds
        generate_wordcloud(df, "Clickbait", "clickbait_wordcloud.png")
        generate_wordcloud(df, "Non-Clickbait", "non_clickbait_wordcloud.png")
        
        # Check if word clouds were created
        if not (os.path.exists("clickbait_wordcloud.png") and os.path.exists("non_clickbait_wordcloud.png")):
            logger.warning("One or more word clouds not generated")
            return False
        
        logger.info("EDA completed successfully")
        return True
    except FileNotFoundError:
        logger.error(f"Dataset {dataset_file} not found")
        return False
    except Exception as e:
        logger.error(f"Error during EDA: {e}")
        return False

if __name__ == "__main__":
    run_eda()