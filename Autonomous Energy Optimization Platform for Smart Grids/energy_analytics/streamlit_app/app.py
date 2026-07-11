import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="EnergyForge Analytics", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    .css-1d391kg, .css-1v3fvcr {
        background-color: #1e2937;
    }
    h1, h2, h3 {
        color: #10b981;
    }
    .stButton>button {
        background-color: #10b981;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #059669;
    }
    .metric-card {
        background-color: #1e2937;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

st.title(" EnergyForge Analytics")
st.markdown("### AI-Powered Smart Energy Management System")
st.markdown("Forecast consumption, uncover patterns, optimize usage.")

@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    daily = pd.read_csv(os.path.join(project_root, 'data/daily_city.csv'))
    acorn = pd.read_csv(os.path.join(project_root, 'data/acorn_daily.csv'))
    daily['datetime'] = pd.to_datetime(daily['datetime'])
    acorn['datetime'] = pd.to_datetime(acorn['datetime'])
    
    daily['month_name'] = daily['datetime'].dt.strftime('%b %Y')
    daily['month_num'] = daily['datetime'].dt.month
    daily['day_of_week'] = daily['datetime'].dt.day_name()
    daily['season'] = daily['month_num'].map(
        {12: 'Winter', 1: 'Winter', 2: 'Winter',
         3: 'Spring',  4: 'Spring',  5: 'Spring',
         6: 'Summer',  7: 'Summer',  8: 'Summer',
         9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
    daily['energy_7d'] = daily['energy_kwh'].rolling(7, min_periods=1).mean()
    daily['energy_30d'] = daily['energy_kwh'].rolling(30, min_periods=1).mean()
    return daily, acorn

daily, acorn = load_data()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Forecasting", "Patterns & Insights", "Optimization"])

if page == "Dashboard":
    st.header("City-Wide Energy Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_cons = daily['energy_kwh'].sum()
        st.metric("Total Avg Consumption (kWh)", f"{total_cons:.0f}", delta=None)
    with col2:
        avg_daily = daily['energy_kwh'].mean()
        st.metric("Avg Daily (kWh)", f"{avg_daily:.1f}")
    with col3:
        peak_daily = daily['energy_kwh'].max()
        st.metric("Peak Daily (kWh)", f"{peak_daily:.1f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['datetime'], y=daily['energy_kwh'],
                             name='Daily', mode='lines',
                             line=dict(color='rgba(16,185,129,0.3)', width=1),
                             fill='tozeroy', fillcolor='rgba(16,185,129,0.05)'))
    fig.add_trace(go.Scatter(x=daily['datetime'], y=daily['energy_7d'],
                             name='7-Day Avg', line=dict(color='#10b981', width=2)))
    fig.add_trace(go.Scatter(x=daily['datetime'], y=daily['energy_30d'],
                             name='30-Day Avg', line=dict(color='#f59e0b', width=2, dash='dot')))
    fig.update_layout(title='Daily Energy Consumption with Rolling Averages',
                      xaxis_title='Date', yaxis_title='Avg kWh',
                      hovermode='x unified', template='plotly_dark',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(daily, x='temperature', y='energy_kwh', color='season',
                      color_discrete_map={'Winter': '#3b82f6', 'Spring': '#10b981',
                                         'Summer': '#f59e0b', 'Autumn': '#ef4444'},
                      title='Consumption vs. Temperature (by Season)',
                      labels={'temperature': 'Temperature (°C)',
                              'energy_kwh': 'Avg kWh', 'season': 'Season'},
                      template='plotly_dark')
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        monthly = daily.groupby(daily['datetime'].dt.to_period('M'))['energy_kwh'].mean().reset_index()
        monthly['period'] = monthly['datetime'].astype(str)
        fig3 = px.bar(monthly, x='period', y='energy_kwh',
                      title='Monthly Average Energy Consumption',
                      labels={'period': 'Month', 'energy_kwh': 'Avg kWh'},
                      color='energy_kwh',
                      color_continuous_scale=['#064e3b', '#10b981', '#6ee7b7'],
                      template='plotly_dark')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = daily.groupby('day_of_week')['energy_kwh'].mean().reindex(dow_order).reset_index()
        dow.columns = ['Day', 'Avg kWh']
        fig4 = px.bar(dow, x='Day', y='Avg kWh',
                      title='Average Consumption by Day of Week',
                      color='Avg kWh',
                      color_continuous_scale=['#064e3b', '#10b981', '#6ee7b7'],
                      template='plotly_dark')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    fig5 = px.box(daily, x='season', y='energy_kwh',
                  category_orders={'season': season_order},
                  color='season',
                  color_discrete_map={'Winter': '#3b82f6', 'Spring': '#10b981',
                                     'Summer': '#f59e0b', 'Autumn': '#ef4444'},
                  title='Energy Consumption Distribution by Season',
                  labels={'season': 'Season', 'energy_kwh': 'Avg kWh'},
                  template='plotly_dark')
    fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

elif page == "Forecasting":
    st.header("Energy Consumption Forecasting")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    model_path = os.path.join(project_root, 'models/arima_daily.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        forecast_steps = 30
        forecast = model.forecast(steps=forecast_steps)
        
        future_dates = pd.date_range(start=daily['datetime'].max() + timedelta(days=1), periods=forecast_steps)
        forecast_df = pd.DataFrame({'datetime': future_dates, 'forecast_kwh': forecast})
        
        st.subheader("30-Day Forecast")
        upper = forecast_df['forecast_kwh'] * 1.10
        lower = forecast_df['forecast_kwh'] * 0.90
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(forecast_df['datetime']) + list(forecast_df['datetime'][::-1]),
            y=list(upper) + list(lower[::-1]),
            fill='toself', fillcolor='rgba(245,158,11,0.1)',
            line=dict(color='rgba(0,0,0,0)'), name='Confidence Band (\u00b110%)'))
        fig.add_trace(go.Scatter(x=daily['datetime'], y=daily['energy_kwh'],
                                 name='Historical', line=dict(color='#10b981', width=2)))
        fig.add_trace(go.Scatter(x=forecast_df['datetime'], y=forecast_df['forecast_kwh'],
                                 name='Forecast', line=dict(color='#f59e0b', width=2.5, dash='dash')))
        fig.update_layout(title='30-Day Energy Forecast with Confidence Band',
                          xaxis_title='Date', yaxis_title='Avg kWh',
                          hovermode='x unified', template='plotly_dark',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        fig_bar = px.bar(forecast_df, x='datetime', y='forecast_kwh',
                         title='Forecasted Daily Consumption (Next 30 Days)',
                         labels={'datetime': 'Date', 'forecast_kwh': 'Forecasted kWh'},
                         color='forecast_kwh',
                         color_continuous_scale=['#064e3b', '#10b981', '#6ee7b7'],
                         template='plotly_dark')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Forecast Data Table")
        st.dataframe(forecast_df.head(10))
    else:
        st.error("Model not found. Train first.")

elif page == "Patterns & Insights":
    st.header("Usage Patterns by ACORN Group")

    fig_acorn = px.line(acorn, x='datetime', y='energy_kwh', color='acorn_group',
                        title='Daily Consumption by ACORN Group',
                        template='plotly_dark')
    fig_acorn.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            hovermode='x unified')
    st.plotly_chart(fig_acorn, use_container_width=True)

    col_p, col_q = st.columns(2)
    with col_p:
        acorn_avg = acorn.groupby('acorn_group')['energy_kwh'].mean().sort_values(ascending=False).reset_index()
        acorn_avg.columns = ['ACORN Group', 'Avg kWh']
        fig_avg = px.bar(acorn_avg, x='ACORN Group', y='Avg kWh',
                         title='Average Consumption per ACORN Group',
                         color='Avg kWh',
                         color_continuous_scale=['#064e3b', '#10b981', '#6ee7b7'],
                         template='plotly_dark')
        fig_avg.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              showlegend=False)
        st.plotly_chart(fig_avg, use_container_width=True)

    with col_q:
        fig_box = px.box(acorn, x='acorn_group', y='energy_kwh', color='acorn_group',
                         title='Consumption Spread by ACORN Group',
                         labels={'acorn_group': 'ACORN Group', 'energy_kwh': 'Avg kWh'},
                         template='plotly_dark')
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    fig_area = px.area(acorn, x='datetime', y='energy_kwh', color='acorn_group',
                       title='Stacked Area Chart — All ACORN Groups Over Time',
                       template='plotly_dark')
    fig_area.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           hovermode='x unified')
    st.plotly_chart(fig_area, use_container_width=True)

    st.subheader("Optimization Insights")
    st.markdown("""
    - **Peak Shifting**: Move usage to off-peak periods.
    - **Temperature Correlation**: Notice how heating/cooling demand drives energy consumption.
    - **Demographic Base Loads**: Different ACORN groups show distinctive base load baselines.
    """)

elif page == "Optimization":
    st.header("Recommendations for Efficiency")
    st.success("Simulated insights: Reduce by 15% through smart scheduling.")

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        season_avg = daily.groupby('season')['energy_kwh'].mean().reindex(season_order).reset_index()
        season_avg.columns = ['Season', 'Avg kWh']
        season_colors = {'Winter': '#3b82f6', 'Spring': '#10b981', 'Summer': '#f59e0b', 'Autumn': '#ef4444'}
        fig_s = px.bar(season_avg, x='Season', y='Avg kWh',
                       color='Season', color_discrete_map=season_colors,
                       title='Average Consumption by Season',
                       template='plotly_dark')
        fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            showlegend=False)
        st.plotly_chart(fig_s, use_container_width=True)

    with col_o2:
        season_total = daily.groupby('season')['energy_kwh'].sum().reindex(season_order).reset_index()
        season_total.columns = ['Season', 'Total kWh']
        fig_pie = px.pie(season_total, values='Total kWh', names='Season',
                         color='Season', color_discrete_map=season_colors,
                         title='Seasonal Share of Total Consumption',
                         hole=0.45, template='plotly_dark')
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                              marker=dict(line=dict(color='#0f172a', width=2)))
        st.plotly_chart(fig_pie, use_container_width=True)

    daily_sorted = daily.sort_values('datetime').copy()
    daily_sorted['cumulative_kwh'] = daily_sorted['energy_kwh'].cumsum()
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=daily_sorted['datetime'], y=daily_sorted['cumulative_kwh'],
        mode='lines', name='Cumulative kWh',
        line=dict(color='#10b981', width=2.5),
        fill='tozeroy', fillcolor='rgba(16,185,129,0.07)'))
    fig_cum.update_layout(title='Cumulative Energy Consumption Over Time',
                          xaxis_title='Date', yaxis_title='Cumulative Avg kWh',
                          template='plotly_dark',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cum, use_container_width=True)

    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    fig_dual.add_trace(go.Scatter(
        x=daily['datetime'], y=daily['energy_30d'],
        name='Energy (30d avg)', line=dict(color='#10b981', width=2),
        fill='tozeroy', fillcolor='rgba(16,185,129,0.05)'), secondary_y=False)
    fig_dual.add_trace(go.Scatter(
        x=daily['datetime'],
        y=daily['temperature'].rolling(30, min_periods=1).mean(),
        name='Temperature (30d avg)', line=dict(color='#f59e0b', width=2)),
        secondary_y=True)
    fig_dual.update_layout(title='Energy Consumption vs Temperature Trend',
                           hovermode='x unified', template='plotly_dark',
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_dual.update_yaxes(title_text='Avg kWh', secondary_y=False)
    fig_dual.update_yaxes(title_text='Temperature (\u00b0C)', secondary_y=True)
    st.plotly_chart(fig_dual, use_container_width=True)

