"""
ui.py
-----
All Streamlit rendering lives here.

Rules:
  - This is the ONLY file allowed to import streamlit.
  - All data work is delegated to data.py; all figure building to charts.py.
  - Functions either render widgets (side effects) or return user-selected
    values collected from widgets.
"""

import datetime
import pandas as pd
import streamlit as st

import data as data_layer
import charts as chart_layer
from config import (
    APP_TITLE, PAGE_LAYOUT,
    CATEGORY_LIST, INCOME_CATEGORIES, PAYMENT_LIST,
    DETAILS_MAX_LENGTH,
)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call in the app
# ---------------------------------------------------------------------------

def configure_page() -> None:
    st.set_page_config(layout=PAGE_LAYOUT, page_title=APP_TITLE)


# ---------------------------------------------------------------------------
# Sidebar — entry form
# ---------------------------------------------------------------------------

def render_entry_form(data: pd.DataFrame) -> pd.DataFrame:
    """
    Render the 'Add New Entry' sidebar form.
    Validates input, handles duplicates, and returns the (possibly updated)
    DataFrame.
    """
    st.sidebar.header('Add New Entry')
    date      = st.sidebar.date_input('Date')
    category  = st.sidebar.selectbox('Category', CATEGORY_LIST)
    amount    = st.sidebar.number_input('Amount', min_value=0, step=1, format='%d')
    payment   = st.sidebar.selectbox('Payment', PAYMENT_LIST)
    details   = st.sidebar.text_area('Details')
    add_entry = st.sidebar.button('Add Entry')
    st.sidebar.markdown('---')

    if not add_entry:
        return data

    # --- Validation ---
    if amount <= 0:
        st.sidebar.error('Amount must be greater than 0.')
        return data

    if len(details) > DETAILS_MAX_LENGTH:
        st.sidebar.error(
            f'Details must be {DETAILS_MAX_LENGTH} characters or fewer '
            f'(currently {len(details)}).'
        )
        return data

    if data_layer.is_duplicate_entry(data, date, category, amount):
        st.sidebar.warning(
            'A similar entry (same date, category, and amount) already exists. '
            'Click Confirm Add to save it anyway.'
        )
        if st.sidebar.button('Confirm Add'):
            data = data_layer.add_new_entry(data, date, category, payment, amount, details)
            data_layer.save_data(data)
            st.sidebar.success('Entry added successfully!')
        return data

    data = data_layer.add_new_entry(data, date, category, payment, amount, details)
    data_layer.save_data(data)
    st.sidebar.success('Entry added successfully!')
    return data


# ---------------------------------------------------------------------------
# Sidebar — date range filter
# ---------------------------------------------------------------------------

def render_date_filter(data: pd.DataFrame):
    """
    Render the date range filter in the sidebar.
    Returns (start_date, end_date) as date objects.

    Defaults:
      - end_date:   today
      - start_date: exactly 1 year before today, clamped to the earliest date
                    in the dataset so the default never falls before any data.
    """
    st.sidebar.header('Filter Data by Date Range')
    today    = datetime.date.today()
    min_date = pd.to_datetime(data['Date']).min().date()

    # 1 year ago today, but never earlier than the oldest record
    one_year_ago   = today.replace(year=today.year - 1)
    default_start  = max(one_year_ago, min_date)

    start_date = st.sidebar.date_input(
        'Start Date',
        value=default_start,
        min_value=min_date,
        max_value=today,
    )
    end_date = st.sidebar.date_input(
        'End Date',
        value=today,
        min_value=min_date,
        max_value=today,
    )
    return start_date, end_date


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def render_summary_metrics(metrics: dict) -> None:
    """
    Render the three top-level metric cards: Income, Expense, Net Balance.

    Args:
        metrics: dict from data_layer.compute_summary_metrics()
    """
    st.markdown('### Summary Overview')
    col1, col2, col3 = st.columns(3)
    col1.metric('Income',  f"${metrics['income']:,}")
    col2.metric('Expense', f"${metrics['expense']:,}")

    savings_rate = metrics['savings_rate']
    delta_str = f"{savings_rate:.1f}% savings rate" if savings_rate is not None else None
    col3.metric('Net Balance', f"${metrics['net_balance']:,}", delta=delta_str)


# ---------------------------------------------------------------------------
# Pivot table
# ---------------------------------------------------------------------------

def render_pivot_table(data: pd.DataFrame) -> None:
    """Build and display the Category × Month pivot table with CSV download."""
    pivot = data_layer.build_pivot_table(data, CATEGORY_LIST)
    st.table(pivot)
    csv = pivot.to_csv().encode('utf-8')
    st.download_button(
        'Download Pivot Table as CSV',
        data=csv, file_name='pivot_table.csv', mime='text/csv',
    )


# ---------------------------------------------------------------------------
# Expense charts
# ---------------------------------------------------------------------------

def render_category_filter() -> list[str]:
    """
    Render category checkboxes (expenses only).
    Returns the list of selected category names.
    """
    expense_categories = [c for c in CATEGORY_LIST if c not in INCOME_CATEGORIES]
    st.subheader('Category Filters')
    cols = st.columns(4)
    selected = {}
    for i, cat in enumerate(expense_categories):
        with cols[i % 4]:
            selected[cat] = st.checkbox(cat, value=True)
    return [c for c in expense_categories if selected[c]]


def render_expense_section(data: pd.DataFrame, selected_categories: list[str]) -> None:
    """Render stacked bar + pie charts for the selected expense categories."""
    if not selected_categories:
        st.warning('Please select at least one category to display charts.')
        return

    monthly, category_summary, sorted_months = data_layer.build_expense_chart_data(
        data, selected_categories
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Monthly Expenses by Category')
        if monthly.empty:
            st.info('No data available for selected categories.')
        else:
            fig = chart_layer.build_expense_bar_chart(monthly, selected_categories, sorted_months)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Category Summary')
        if category_summary.empty:
            st.info('No data available for selected categories.')
        else:
            fig = chart_layer.build_expense_pie_chart(category_summary, selected_categories)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Income vs Expense  +  Savings Rate
# ---------------------------------------------------------------------------

def render_income_expense_section(data: pd.DataFrame) -> None:
    """Render the grouped bar chart with trend lines."""
    st.subheader('Monthly Income vs Expenses')
    monthly_summary = data_layer.build_monthly_summary(data)
    fig = chart_layer.build_income_expense_chart(monthly_summary)
    st.plotly_chart(fig, use_container_width=True)


def render_savings_rate_section(data: pd.DataFrame) -> None:
    """Render the savings rate line chart."""
    monthly_summary = data_layer.build_monthly_summary(data)
    valid = monthly_summary.dropna(subset=['SavingsRate'])
    if valid.empty:
        st.info('No income data available to compute savings rate.')
        return
    st.subheader('Monthly Savings Rate')
    fig = chart_layer.build_savings_rate_chart(monthly_summary)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Payment method breakdown
# ---------------------------------------------------------------------------

def render_payment_section(data: pd.DataFrame) -> None:
    """Render stacked bar + donut charts for payment method breakdown."""
    monthly_pay, pay_summary, sorted_months, present_payments = (
        data_layer.build_payment_chart_data(data, PAYMENT_LIST)
    )
    if not present_payments:
        st.info('No expense data to show payment breakdown.')
        return

    st.subheader('Payment Method Breakdown')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Monthly Spending by Payment Method**')
        fig = chart_layer.build_payment_bar_chart(monthly_pay, present_payments, sorted_months)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('**Overall Payment Method Share**')
        fig = chart_layer.build_payment_donut_chart(pay_summary, present_payments)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Year-over-Year comparison
# ---------------------------------------------------------------------------

def render_yoy_section(data: pd.DataFrame) -> None:
    """
    Render the year-over-year comparison chart.
    Silently skipped when data spans fewer than 2 calendar years.
    """
    years = pd.to_datetime(data['Date']).dt.year.unique()
    if len(years) < 2:
        return  # not enough data yet — no error shown

    st.subheader('Year-over-Year Comparison')
    view = st.radio('Compare by:', ['Expense', 'Income'], horizontal=True, key='yoy_view')

    monthly_yoy, _ = data_layer.build_yoy_data(data, view)
    fig = chart_layer.build_yoy_chart(monthly_yoy, view)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Recent entries  (edit / delete)
# ---------------------------------------------------------------------------

def render_recent_entries(data: pd.DataFrame) -> None:
    """
    Display the five most recent entries with inline edit and delete controls.
    Saves and triggers a rerun when any change is committed.
    """
    st.subheader('Recent Entries')
    if data.empty:
        st.info('No entries available.')
        return

    recent = data.sort_values('Date', ascending=False).head(5).copy()
    recent['Date'] = pd.to_datetime(recent['Date'])
    st.markdown('### Most Recent Entries (Edit or Delete)')
    updated = False

    for idx, row in recent.iterrows():
        label = f"{row['Date'].date()} | {row['Category']} | ${row['Amount']}"
        with st.expander(label):
            new_date     = st.date_input('Date', row['Date'].date(), key=f'date_{idx}')
            new_category = st.selectbox(
                'Category', CATEGORY_LIST,
                index=CATEGORY_LIST.index(row['Category']), key=f'cat_{idx}',
            )
            new_payment  = st.selectbox(
                'Payment', PAYMENT_LIST,
                index=PAYMENT_LIST.index(row['Payment']), key=f'pay_{idx}',
            )
            new_amount   = st.number_input(
                'Amount', min_value=0, value=int(row['Amount']), step=1, key=f'amt_{idx}',
            )
            new_details  = st.text_input('Details', row['Details'], key=f'det_{idx}')

            c1, c2 = st.columns(2)
            with c1:
                if st.button('Save Changes', key=f'save_{idx}'):
                    data.loc[idx, 'Date']     = pd.to_datetime(new_date)
                    data.loc[idx, 'Category'] = new_category
                    data.loc[idx, 'Payment']  = new_payment
                    data.loc[idx, 'Amount']   = int(new_amount)
                    data.loc[idx, 'Details']  = new_details
                    updated = True
            with c2:
                if st.button('Delete Entry', key=f'del_{idx}'):
                    data    = data[data.index != idx].reset_index(drop=True)
                    updated = True

    if updated:
        data_layer.save_data(data)
        st.success('Changes applied.')
        st.rerun()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    configure_page()
    st.title(APP_TITLE)

    data = data_layer.load_data()

    # --- Sidebar: add entry ---
    data = render_entry_form(data)

    if data.empty:
        st.info('No data available. Please add some entries to get started.')
        return

    # --- Sidebar: date filter ---
    start_date, end_date = render_date_filter(data)
    filtered = data_layer.filter_data_by_date(data, start_date, end_date)

    if filtered.empty:
        st.warning('No data available for the selected date range.')
        return

    # --- Date range caption ---
    st.caption(
        f"Showing data for selected date range "
        f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
    )

    # --- Summary metrics ---
    metrics = data_layer.compute_summary_metrics(filtered)
    render_summary_metrics(metrics)

    # --- Pivot table ---
    render_pivot_table(filtered)

    # --- Expense charts ---
    selected_categories = render_category_filter()
    render_expense_section(filtered, selected_categories)

    # --- Income vs Expense + trend lines ---
    render_income_expense_section(filtered)

    # --- Savings rate ---
    render_savings_rate_section(filtered)

    # --- Payment breakdown ---
    render_payment_section(filtered)

    # --- Year-over-year (uses full dataset — independent of date filter) ---
    render_yoy_section(data)

    # --- Recent entries (always uses full unfiltered dataset) ---
    render_recent_entries(data)