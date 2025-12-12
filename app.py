import streamlit as st

# Import init_db from your source file to ensure tables are created on startup
from src.database import init_db, get_db_counts

# Initialize database connection and tables
init_db()

# --- Page Configuration ---
# Set page title to "Home" - this is the title that appears in the browser tab
st.set_page_config(
    layout="wide",
    page_title="Home | Stock Analysis",
    menu_items={'About': "A simple financial analysis app using Streamlit and DuckDB."}
)

st.title("🏠 Home: Application Overview")
st.markdown("---")

st.header("Quick Recap & Next Steps")
st.markdown(
    """
    Welcome to the Stock Analysis application, your dashboard for French market data. 
    The application is organized into the following sections, accessible via the sidebar navigation:
    """
)

# Use columns for a cleaner, dual-purpose recap
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 Analysis Dashboard")
    st.markdown(
        """
        * **Purpose:** Detailed analysis of selected tickers.
        * **Features:** View historical price charts, technical indicators (e.g., SMA, RSI), and comparative performance against other stocks.
        * **Next Action:** Select a ticker and analysis period to begin your financial review.
        """
    )

with col2:
    st.subheader("⚙️ Data Management")
    st.markdown(
        """
        * **Purpose:** Database maintenance and data ingestion.
        * **Features:** Load base tickers from the local CSV, fetch/update historical pricing data efficiently, and manually add new tickers.
        * **⚠️ Critical Step:** **Before analysis**, please check this page to ensure your database is populated and up-to-date.
        """
    )

st.markdown("---")

# Optional: Add a simple metric about the data status
# We'll use the functions we built to show a quick status, enhancing the "recap" nature.
from src.database import get_db_connection, get_ticker_details

conn = get_db_connection()

try:
    # Safely get counts
    ticker_count, historical_count, _ = get_db_counts()

    st.header("Current Data Status")

    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.metric("Total Tickers in DB", value=ticker_count)

    with status_col2:
        st.metric("Total Historical Data Points", value=f"{historical_count:,}")

except Exception as e:
    st.warning(f"Could not load data status: Ensure the database has been initialized.")