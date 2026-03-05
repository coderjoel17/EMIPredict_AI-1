"""
Data Analysis and Visualization Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

st.title("📊 Data Analysis & Insights")
st.markdown("### Explore patterns and trends in EMI applications")

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/processed/train.csv')
    except:
        return None

df = load_data()

# Ensure required columns exist to prevent KeyErrors
required_defaults = {
    'emi_scenario': 'Standard',
    'employment_type': 'Salaried',
    'requested_amount': 0,
    'requested_tenure': 24,
    'current_emi_amount': 0,
    'monthly_rent': 0,
    'school_fees': 0,
    'college_fees': 0,
    'travel_expenses': 0,
    'groceries_utilities': 0,
    'other_monthly_expenses': 0
}

for col, default in required_defaults.items():
    if col not in df.columns:
        df[col] = default

if df is None:
    st.error("⚠️ Data not found! Please run preprocessing first.")
    st.stop()

# --- Ensure numeric columns are properly converted ---
numeric_cols = [
    'age', 'monthly_salary', 'credit_score', 'bank_balance',
    'emergency_fund', 'current_emi_amount', 'max_monthly_emi',
    'requested_amount', 'requested_tenure', 'monthly_rent',
    'school_fees', 'college_fees', 'travel_expenses',
    'groceries_utilities', 'other_monthly_expenses'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')



# Sidebar filters
st.sidebar.markdown("## 🎛️ Filters")
emi_scenarios = ['All'] + list(df['emi_scenario'].unique())
selected_scenario = st.sidebar.selectbox("EMI Scenario", emi_scenarios)

if selected_scenario != 'All':
    df_filtered = df[df['emi_scenario'] == selected_scenario]
else:
    df_filtered = df.copy()

# Overview metrics
st.markdown("---")
st.markdown("## 📈 Overview Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Applications", f"{len(df_filtered):,}")
with col2:
    eligible_pct = (df_filtered['emi_eligibility'] == 'Eligible').sum() / len(df_filtered) * 100
    st.metric("Eligible Applications", f"{eligible_pct:.1f}%")
with col3:
    avg_salary = df_filtered['monthly_salary'].mean()
    st.metric("Average Salary", f"₹{avg_salary:,.0f}")
with col4:
    avg_credit = df_filtered['credit_score'].mean()
    st.metric("Average Credit Score", f"{avg_credit:.0f}")

# Visualizations
st.markdown("---")

# Row 1: Eligibility Distribution and EMI Distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Eligibility Distribution")
    eligibility_counts = df_filtered['emi_eligibility'].value_counts()
    fig = px.pie(values=eligibility_counts.values, 
                 names=eligibility_counts.index,
                 color=eligibility_counts.index,
                 color_discrete_map={
                     'Eligible': '#28a745',
                     'High_Risk': '#ffc107',
                     'Not_Eligible': '#dc3545'
                 })
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, width="stretch")

with col2:
    st.markdown("### 💰 Max Monthly EMI Distribution")
    fig = px.histogram(df_filtered, x='max_monthly_emi', nbins=50,
                      color_discrete_sequence=['#1f77b4'])
    fig.update_layout(xaxis_title="Max Monthly EMI (₹)", 
                     yaxis_title="Count",
                     showlegend=False)
    st.plotly_chart(fig, width="stretch")

# Row 2: Age and Salary Analysis
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👥 Age Distribution by Eligibility")
    fig = px.box(df_filtered, x='emi_eligibility', y='age',
                color='emi_eligibility',
                color_discrete_map={
                    'Eligible': '#28a745',
                    'High_Risk': '#ffc107',
                    'Not_Eligible': '#dc3545'
                })
    fig.update_layout(xaxis_title="Eligibility Status", 
                     yaxis_title="Age",
                     showlegend=False)
    st.plotly_chart(fig, width="stretch")

with col2:
    st.markdown("### 💵 Monthly Salary by Eligibility")
    fig = px.box(df_filtered, x='emi_eligibility', y='monthly_salary',
                color='emi_eligibility',
                color_discrete_map={
                    'Eligible': '#28a745',
                    'High_Risk': '#ffc107',
                    'Not_Eligible': '#dc3545'
                })
    fig.update_layout(xaxis_title="Eligibility Status", 
                     yaxis_title="Monthly Salary (₹)",
                     showlegend=False)
    st.plotly_chart(fig, width="stretch")

# Row 3: Credit Score and Employment Analysis
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Credit Score Distribution")
    fig = px.violin(df_filtered, x='emi_eligibility', y='credit_score',
                   color='emi_eligibility', box=True,
                   color_discrete_map={
                       'Eligible': '#28a745',
                       'High_Risk': '#ffc107',
                       'Not_Eligible': '#dc3545'
                   })
    fig.update_layout(xaxis_title="Eligibility Status", 
                     yaxis_title="Credit Score",
                     showlegend=False)
    st.plotly_chart(fig, width="stretch")

with col2:
    st.markdown("### 💼 Employment Type Analysis")
    emp_analysis = pd.crosstab(df_filtered['employment_type'], 
                               df_filtered['emi_eligibility'], 
                               normalize='index') * 100
    
    fig = go.Figure()
    for col in emp_analysis.columns:
        fig.add_trace(go.Bar(name=col, x=emp_analysis.index, y=emp_analysis[col]))
    
    fig.update_layout(barmode='stack',
                     xaxis_title="Employment Type",
                     yaxis_title="Percentage (%)",
                     legend_title="Eligibility")
    st.plotly_chart(fig, width="stretch")

# Row 4: EMI Scenario Analysis
st.markdown("---")
st.markdown("### 🎯 EMI Scenario Comparison")

agg_dict = {
    'max_monthly_emi': 'mean'
}

if 'requested_amount' in df.columns:
    agg_dict['requested_amount'] = ['mean', 'median']

if 'requested_tenure' in df.columns:
    agg_dict['requested_tenure'] = 'mean'

# Create scenario statistics
scenario_stats = df.groupby('emi_scenario').agg(agg_dict).round(2)

# Flatten multi-level column names
scenario_stats.columns = ['_'.join(col).strip('_') for col in scenario_stats.columns]

# Calculate approval rate
approval_rate = df.groupby('emi_scenario')['emi_eligibility'] \
    .apply(lambda x: (x == 'Eligible').sum() / len(x) * 100)

scenario_stats['Approval_Rate'] = approval_rate

# Reset index
scenario_stats = scenario_stats.reset_index()

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(scenario_stats, x='emi_scenario', y='requested_amount_mean',
                color='requested_amount_mean', color_continuous_scale='Blues')
    fig.update_layout(xaxis_title="EMI Scenario",
                     yaxis_title="Average Requested Amount (₹)",
                     showlegend=False)
    st.plotly_chart(fig, width="stretch")

with col2:
    fig = px.bar(scenario_stats, x='emi_scenario', y='Approval_Rate',
                color='Approval_Rate', color_continuous_scale='Greens')
    fig.update_layout(xaxis_title="EMI Scenario",
                     yaxis_title="Approval Rate (%)",
                     showlegend=False)
    st.plotly_chart(fig, width="stretch")

# Detailed Statistics Table
st.markdown("---")
st.markdown("### 📋 Detailed Statistics by Scenario")
st.dataframe(scenario_stats, use_container_width=True)

# Financial Metrics Analysis
st.markdown("---")
st.markdown("### 💸 Financial Metrics Analysis")

# Calculate derived metrics
expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
               'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
existing_expense_cols = [c for c in expense_cols if c in df_filtered.columns]
df_filtered['total_expenses'] = df_filtered[existing_expense_cols].sum(axis=1)
df_filtered['debt_to_income'] = (df_filtered['current_emi_amount'] / df_filtered['monthly_salary'] * 100).clip(0, 100)
df_filtered['expense_ratio'] = (df_filtered['total_expenses'] / df_filtered['monthly_salary'] * 100).clip(0, 150)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Debt-to-Income Ratio")
    fig = px.box(df_filtered, x='emi_eligibility', y='debt_to_income',
                color='emi_eligibility',
                color_discrete_map={
                    'Eligible': '#28a745',
                    'High_Risk': '#ffc107',
                    'Not_Eligible': '#dc3545'
                })
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="DTI Ratio (%)")
    st.plotly_chart(fig, width="stretch")

with col2:
    st.markdown("#### Expense Ratio")
    fig = px.box(df_filtered, x='emi_eligibility', y='expense_ratio',
                color='emi_eligibility',
                color_discrete_map={
                    'Eligible': '#28a745',
                    'High_Risk': '#ffc107',
                    'Not_Eligible': '#dc3545'
                })
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Expense Ratio (%)")
    st.plotly_chart(fig, width="stretch")

with col3:
    st.markdown("#### Bank Balance")
    fig = px.box(df_filtered, x='emi_eligibility', y='bank_balance',
                color='emi_eligibility',
                color_discrete_map={
                    'Eligible': '#28a745',
                    'High_Risk': '#ffc107',
                    'Not_Eligible': '#dc3545'
                })
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Balance (₹)")
    st.plotly_chart(fig, width="stretch")

# Key Insights
st.markdown("---")
st.markdown("### 💡 Key Insights")

eligible_df = df_filtered[df_filtered['emi_eligibility'] == 'Eligible']
not_eligible_df = df_filtered[df_filtered['emi_eligibility'] == 'Not_Eligible']

col1, col2 = st.columns(2)

with col1:
    st.success("#### ✅ Eligible Applicants Profile")
    st.markdown(f"""
    - **Average Salary**: ₹{eligible_df['monthly_salary'].mean():,.0f}
    - **Average Credit Score**: {eligible_df['credit_score'].mean():.0f}
    - **Average Bank Balance**: ₹{eligible_df['bank_balance'].mean():,.0f}
    - **Average Debt-to-Income**: {eligible_df['debt_to_income'].mean():.1f}%
    - **Average Age**: {eligible_df['age'].mean():.1f} years
    """)

with col2:
    st.error("#### ❌ Not Eligible Applicants Profile")
    st.markdown(f"""
    - **Average Salary**: ₹{not_eligible_df['monthly_salary'].mean():,.0f}
    - **Average Credit Score**: {not_eligible_df['credit_score'].mean():.0f}
    - **Average Bank Balance**: ₹{not_eligible_df['bank_balance'].mean():,.0f}
    - **Average Debt-to-Income**: {not_eligible_df['debt_to_income'].mean():.1f}%
    - **Average Age**: {not_eligible_df['age'].mean():.1f} years
    """)

# Correlation heatmap
st.markdown("---")
st.markdown("### 🔗 Feature Correlations")

numerical_cols = ['age', 'monthly_salary', 'credit_score', 'bank_balance', 
                 'current_emi_amount', 'requested_amount', 'max_monthly_emi']
existing_num_cols = [c for c in numerical_cols if c in df_filtered.columns]
corr_matrix = df_filtered[existing_num_cols].corr()

fig = px.imshow(corr_matrix, 
                text_auto=True, 
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1)
fig.update_layout(title="Correlation Matrix", height=600)
st.plotly_chart(fig, width="stretch")

# Download section
st.markdown("---")
st.markdown("### 📥 Export Analysis")

csv = df_filtered.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name=f"emi_analysis_{selected_scenario}.csv",
    mime="text/csv"
)