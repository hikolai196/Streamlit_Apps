# charts.py
# All chart-building logic
# Pure chart-building functions.

# Rules:
#   - Zero Streamlit imports.
#   - Every function receives pre-processed data and returns a Plotly Figure.
#   - No direct data aggregation — callers (ui.py) pass data already prepared
#     by data.py helpers.

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config import CATEGORY_COLORS, TYPE_COLORS, PAYMENT_COLORS

# ---------------------------------------------------------------------------
# Expense charts
# ---------------------------------------------------------------------------

def build_expense_bar_chart(
    monthly: pd.DataFrame,
    filtered_categories: list[str],
    sorted_months: list,
) -> go.Figure:
    """
    Stacked bar chart: monthly expense totals broken down by category.

    Args:
        monthly:             DataFrame with columns Month-Year, Category, Amount
        filtered_categories: ordered list of categories to include
        sorted_months:       month labels in chronological order (for x-axis)
    """
    fig = px.bar(
        monthly,
        x='Month-Year', y='Amount', color='Category',
        barmode='stack',
        color_discrete_map={c: CATEGORY_COLORS[c] for c in filtered_categories},
        category_orders={
            'Category':   filtered_categories,
            'Month-Year': sorted_months,
        },
    )
    fig.update_layout(xaxis_title='Month-Year', yaxis_title='Amount')
    return fig


def build_expense_pie_chart(
    category_summary: pd.DataFrame,
    filtered_categories: list[str],
) -> go.Figure:
    """
    Pie chart: total spending share per category over the selected period.

    Args:
        category_summary:    DataFrame with columns Category, Amount
        filtered_categories: ordered list of categories (controls legend order)
    """
    fig = px.pie(
        category_summary,
        values='Amount', names='Category',
        color='Category',
        color_discrete_map={c: CATEGORY_COLORS[c] for c in filtered_categories},
        category_orders={'Category': filtered_categories},
    )
    return fig


# ---------------------------------------------------------------------------
# Income vs Expense chart  (with trend lines — improvement #14)
# ---------------------------------------------------------------------------

def build_income_expense_chart(monthly_summary: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart for monthly Income vs Expense, with 3-month rolling
    average trend lines overlaid for each series.

    Args:
        monthly_summary: output of data.build_monthly_summary()
                         (columns: Month-Year, Income, Expense)
    """
    sorted_months  = monthly_summary['Month-Year'].tolist()
    income_trend   = monthly_summary['Income'].rolling(window=3, min_periods=1).mean().round(0)
    expense_trend  = monthly_summary['Expense'].rolling(window=3, min_periods=1).mean().round(0)

    fig = go.Figure()

    # Bars
    fig.add_trace(go.Bar(
        x=monthly_summary['Month-Year'], y=monthly_summary['Income'],
        name='Income', marker_color=TYPE_COLORS['Income'],
        text=monthly_summary['Income'], textposition='outside',
    ))
    fig.add_trace(go.Bar(
        x=monthly_summary['Month-Year'], y=monthly_summary['Expense'],
        name='Expense', marker_color=TYPE_COLORS['Expense'],
        text=monthly_summary['Expense'], textposition='outside',
    ))

    # Trend lines
    fig.add_trace(go.Scatter(
        x=monthly_summary['Month-Year'], y=income_trend,
        name='Income trend (3-mo avg)', mode='lines+markers',
        line=dict(color='#1a5c38', width=2, dash='dot'),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=monthly_summary['Month-Year'], y=expense_trend,
        name='Expense trend (3-mo avg)', mode='lines+markers',
        line=dict(color='#7ab0b0', width=2, dash='dot'),
        marker=dict(size=5),
    ))

    fig.update_layout(
        barmode='group',
        xaxis=dict(
            title='Month-Year',
            categoryorder='array',
            categoryarray=sorted_months,
        ),
        yaxis_title='Amount',
        legend_title='',
    )
    return fig


# ---------------------------------------------------------------------------
# Savings rate chart  (improvement #15)
# ---------------------------------------------------------------------------

def build_savings_rate_chart(monthly_summary: pd.DataFrame) -> go.Figure:
    """
    Filled line chart of monthly savings rate (%).
    Months without income are excluded (rate is undefined).
    A dashed break-even line is drawn at 0%.

    Args:
        monthly_summary: output of data.build_monthly_summary()
                         (columns: Month-Year, SavingsRate)
    """
    valid = monthly_summary.dropna(subset=['SavingsRate'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid['Month-Year'],
        y=valid['SavingsRate'].round(1),
        mode='lines+markers+text',
        text=valid['SavingsRate'].round(1).astype(str) + '%',
        textposition='top center',
        line=dict(color='#2E8B57', width=2),
        fill='tozeroy',
        fillcolor='rgba(46,139,87,0.12)',
    ))
    fig.add_hline(
        y=0, line_dash='dash', line_color='gray',
        annotation_text='Break-even (0%)',
        annotation_position='bottom right',
    )
    fig.update_layout(
        xaxis=dict(
            title='Month-Year',
            categoryorder='array',
            categoryarray=valid['Month-Year'].tolist(),
        ),
        yaxis_title='Savings Rate (%)',
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Payment method charts  (improvement #16)
# ---------------------------------------------------------------------------

def build_payment_bar_chart(
    monthly_pay: pd.DataFrame,
    present_payments: list[str],
    sorted_months: list,
) -> go.Figure:
    """
    Stacked bar chart: monthly expense totals broken down by payment method.

    Args:
        monthly_pay:      DataFrame with columns Month-Year, Payment, Amount
        present_payments: payment methods actually present in the data slice
        sorted_months:    month labels in chronological order
    """
    fig = px.bar(
        monthly_pay,
        x='Month-Year', y='Amount', color='Payment',
        barmode='stack',
        color_discrete_map={p: PAYMENT_COLORS[p] for p in present_payments},
        category_orders={
            'Payment':    present_payments,
            'Month-Year': sorted_months,
        },
    )
    fig.update_layout(xaxis_title='Month-Year', yaxis_title='Amount')
    return fig


def build_payment_donut_chart(
    pay_summary: pd.DataFrame,
    present_payments: list[str],
) -> go.Figure:
    """
    Donut chart: overall spending share per payment method.

    Args:
        pay_summary:      DataFrame with columns Payment, Amount
        present_payments: payment methods actually present in the data slice
    """
    fig = px.pie(
        pay_summary,
        values='Amount', names='Payment',
        hole=0.4,
        color='Payment',
        color_discrete_map={p: PAYMENT_COLORS[p] for p in present_payments},
        category_orders={'Payment': present_payments},
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


# ---------------------------------------------------------------------------
# Year-over-year chart  (improvement #17)
# ---------------------------------------------------------------------------

def build_yoy_chart(
    monthly_yoy: pd.DataFrame,
    view: str,
) -> go.Figure:
    """
    Line chart with months (Jan–Dec) on the X-axis and one line per year.

    Args:
        monthly_yoy: output of data.build_yoy_data() —
                     columns: Year (str), Month, MonthLabel, Amount
        view:        'Expense' or 'Income' (used only for axis label)
    """
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = px.line(
        monthly_yoy,
        x='MonthLabel', y='Amount', color='Year',
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set2,
        category_orders={'MonthLabel': month_order},
        labels={
            'MonthLabel': 'Month',
            'Amount':     f'Total {view}',
            'Year':       'Year',
        },
    )
    fig.update_traces(line=dict(width=2), marker=dict(size=7))
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title=f'Total {view}',
        legend_title='Year',
    )
    return fig