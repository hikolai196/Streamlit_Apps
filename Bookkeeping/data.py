# data.py
# All CSV/data processing logic

# Pure data-layer functions: loading, saving, querying, and transforming
# the bookkeeping CSV.

# Rules:
#   - Zero Streamlit imports.
#   - Every function takes plain Python / pandas values and returns plain values.
#   - No side effects beyond reading/writing CSV_FILE.

import pandas as pd
from config import CSV_FILE, INCOME_CATEGORIES

# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load bookkeeping data from CSV.
    Returns an empty DataFrame with correct columns if the file does not exist.
    """
    import os
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, parse_dates=['Date'])
        df['Date']   = pd.to_datetime(df['Date'])
        df['Amount'] = df['Amount'].astype(int)
        return df
    return pd.DataFrame(columns=['Date', 'Category', 'Payment', 'Amount', 'Details'])


def save_data(data: pd.DataFrame) -> None:
    """Persist DataFrame to CSV. Dates are serialised as ISO strings."""
    out = data.copy()
    out['Date'] = out['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    out.to_csv(CSV_FILE, index=False)


def add_new_entry(
    data: pd.DataFrame,
    date,
    category: str,
    payment: str,
    amount: int,
    details: str,
) -> pd.DataFrame:
    """Append a new row and return the updated DataFrame (original unchanged)."""
    new_row = pd.DataFrame({
        'Date':     [pd.to_datetime(date)],
        'Category': [category],
        'Payment':  [payment],
        'Amount':   [int(amount)],
        'Details':  [details],
    })
    return pd.concat([data, new_row], ignore_index=True)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_duplicate_entry(
    data: pd.DataFrame,
    date,
    category: str,
    amount: int,
) -> bool:
    """
    Return True if an entry with the same date (date portion only),
    category, and amount already exists.
    """
    if data.empty:
        return False
    date_only = pd.to_datetime(date).date()
    match = data[
        (pd.to_datetime(data['Date']).dt.date == date_only) &
        (data['Category'] == category) &
        (data['Amount']   == int(amount))
    ]
    return not match.empty


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_data_by_date(
    data: pd.DataFrame,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Return rows where Date falls within [start_date, end_date] inclusive.
    end_date is normalised to end-of-day so entries with non-midnight
    timestamps on the end date are not excluded.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    return data[(data['Date'] >= start_dt) & (data['Date'] < end_dt)]


# ---------------------------------------------------------------------------
# Aggregation / transformation
# ---------------------------------------------------------------------------

def build_monthly_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data to one row per calendar month with columns:
        Month-Year-Obj  (Timestamp — for correct chronological sorting)
        Month-Year      (str label, e.g. 'Jan-24')
        Income          (int)
        Expense         (int)
        Net             (int)
        SavingsRate     (float | None — None when Income == 0)

    Used by multiple chart builders to avoid duplicating pivot logic.
    """
    df = data.copy()
    df['Month-Year-Obj'] = (
        pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
    )
    df['Type'] = df['Category'].apply(
        lambda x: 'Income' if x in INCOME_CATEGORIES else 'Expense'
    )
    pivot = (
        df.groupby(['Month-Year-Obj', 'Type'])['Amount']
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Ensure both columns exist even if only one type appears in the range
    for col in ('Income', 'Expense'):
        if col not in pivot.columns:
            pivot[col] = 0

    pivot['Net']         = pivot['Income'] - pivot['Expense']
    pivot['SavingsRate'] = pivot.apply(
        lambda r: (r['Net'] / r['Income'] * 100) if r['Income'] > 0 else None,
        axis=1,
    )
    pivot['Month-Year'] = pivot['Month-Year-Obj'].dt.strftime('%b-%y')
    return pivot.sort_values('Month-Year-Obj').reset_index(drop=True)


def build_pivot_table(data: pd.DataFrame, category_list: list) -> pd.DataFrame:
    """
    Build a Category × Month-Year pivot table for display.
    Rows are ordered according to category_list.
    """
    df = data.copy()
    df['Month-Year'] = pd.to_datetime(df['Date']).dt.to_period('M')
    pivot = df.pivot_table(
        index='Category', columns='Month-Year',
        values='Amount', aggfunc='sum', fill_value=0,
    )
    pivot.columns = pivot.columns.strftime('%b-%y')
    return pivot.reindex(category_list)


def compute_summary_metrics(data: pd.DataFrame) -> dict:
    """
    Return a dict with keys: income, expense, net_balance, savings_rate (or None).
    Centralises the income/expense split so ui.py does not need to know about
    INCOME_CATEGORIES directly.
    """
    income      = int(data[data['Category'].isin(INCOME_CATEGORIES)]['Amount'].sum())
    expense     = int(data[~data['Category'].isin(INCOME_CATEGORIES)]['Amount'].sum())
    net_balance = income - expense
    savings_rate = round((net_balance / income * 100), 1) if income > 0 else None
    return {
        'income':       income,
        'expense':      expense,
        'net_balance':  net_balance,
        'savings_rate': savings_rate,
    }


def build_expense_chart_data(
    data: pd.DataFrame,
    filtered_categories: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Prepare data for the expense bar + pie charts.

    Returns:
        monthly_summary   — grouped by Month-Year × Category
        category_summary  — grouped by Category (for pie)
        sorted_months     — month labels in chronological order
    """
    df = data[data['Category'].isin(filtered_categories)].copy()

    df['Month-Year-Obj'] = (
        pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
    )
    df['Month-Year'] = df['Month-Year-Obj'].dt.strftime('%b-%y')

    monthly = (
        df.sort_values('Month-Year-Obj')
        .groupby(['Month-Year', 'Category'])['Amount']
        .sum()
        .reset_index()
    )
    category_summary = df.groupby('Category')['Amount'].sum().reset_index()
    sorted_months    = df.sort_values('Month-Year-Obj')['Month-Year'].unique().tolist()

    return monthly, category_summary, sorted_months


def build_payment_chart_data(
    data: pd.DataFrame,
    payment_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """
    Prepare data for the payment method bar + donut charts (expenses only).

    Returns:
        monthly_pay       — grouped by Month-Year × Payment
        pay_summary       — grouped by Payment (for donut)
        sorted_months     — month labels in chronological order
        present_payments  — payment methods actually found in this data slice
    """
    df = data[~data['Category'].isin(INCOME_CATEGORIES)].copy()

    df['Month-Year-Obj'] = (
        pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
    )
    df['Month-Year'] = df['Month-Year-Obj'].dt.strftime('%b-%y')

    monthly_pay = (
        df.sort_values('Month-Year-Obj')
        .groupby(['Month-Year', 'Payment'])['Amount']
        .sum()
        .reset_index()
    )
    pay_summary      = df.groupby('Payment')['Amount'].sum().reset_index()
    sorted_months    = df.sort_values('Month-Year-Obj')['Month-Year'].unique().tolist()
    present_payments = [p for p in payment_list if p in df['Payment'].unique()]

    return monthly_pay, pay_summary, sorted_months, present_payments


def build_yoy_data(
    data: pd.DataFrame,
    view: str,
) -> tuple[pd.DataFrame, list]:
    """
    Prepare data for the year-over-year comparison chart.

    Args:
        view: 'Expense' or 'Income'

    Returns:
        monthly_yoy  — Year × MonthLabel aggregation
        years        — sorted list of calendar years present (as str)
    """
    df = data.copy()
    df['Year']       = pd.to_datetime(df['Date']).dt.year
    df['Month']      = pd.to_datetime(df['Date']).dt.month
    df['MonthLabel'] = pd.to_datetime(df['Date']).dt.strftime('%b')

    years = sorted(df['Year'].unique())

    df_view = (
        df[~df['Category'].isin(INCOME_CATEGORIES)]
        if view == 'Expense'
        else df[df['Category'].isin(INCOME_CATEGORIES)]
    )

    monthly_yoy = (
        df_view
        .groupby(['Year', 'Month', 'MonthLabel'])['Amount']
        .sum()
        .reset_index()
        .sort_values(['Year', 'Month'])
    )
    monthly_yoy['Year'] = monthly_yoy['Year'].astype(str)

    return monthly_yoy, [str(y) for y in years]