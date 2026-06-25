# Bookkeeping App

A streamlined, interactive dashboard built with **Streamlit** to manage and visualize personal or small business finances. This application transforms raw CSV data into actionable insights through intuitive visualizations and automated summaries.

## 🚀 Features
- **Data Management**: Easy integration of CSV files for tracking income and expenses.
- **Dynamic Filtering**: View financial records over specific date ranges (e.g., Monthly, Quarterly).
- **Advanced Analytics**:
    - **Monthly Summaries**: Automatically calculate net balance and savings rates.
    - **Expense Breakdown**: Visualize spending across different categories and payment methods via interactive charts.
    - **Year-over-Year (YoY) Comparison**: Compare current performance against previous periods to spot trends.
- **Modular Architecture**: Clean separation of concerns between data processing (`data.py`), visualization logic (`charts.py`), and the user interface (`ui.py`).

## 🛠️ Tech Stack
- **Python**
- **Streamlit** (Web Interface)
- **Pandas** (Data Manipulation)
- **Plotly** (Interactive Visualizations)
- **uv** (High-performance Python package management)

## 🚀 Getting Started

### Installation

1. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

2. Run the application:
   ```bash
   uv run streamlit run bookkeeping_app.py
   ```

### For those using pip and venv

1. Install dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run bookkeeping_app.py
   ```

## 📂 Project Structure
- `bookkeeping_app.py`: Main entry point for the Streamlit application.
- `ui.py`: Handles layout, widgets, and navigation.
- `data.py`: Contains all logic for CSV loading, data validation, filtering, and aggregation.
- `charts.py`: Logic for constructing Plotly figures and custom chart components.
- `config.py`: Configuration constants (e.g., income categories) and environment settings.

## 📖 Usage Guide
1. **Data Input**: The system reads from a CSV file defined in the configuration.
2. **Navigation**: Use the sidebar or tabs to switch between "Dashboard", "Expenses", and "Payments".
3. **Customization**: Adjust filters to see specific timeframes or filtered categories.

## 📄 License
[Insert your license type, e.g., MIT]