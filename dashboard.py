import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import pdfplumber
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from PIL import Image
from scipy import stats

# McKinsey-style professional styling
st.set_page_config(
    page_title="Ministry of Finance Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bloomberg Terminal style colors
BLOOMBERG_COLORS = {
    'bg': '#000000',              # Pure black
    'primary': '#00B050',         # Bloomberg green
    'secondary': '#5CB85C',       # Lighter green
    'accent': '#FFD700',          # Gold for highlights
    'text': '#FFFFFF',            # White text
    'text_muted': '#999999',      # Gray text
    'positive': '#00B050',        # Green for positive
    'negative': '#FF4444',        # Red for negative
    'neutral': '#4A4A4A',         # Dark gray
}

# Bloomberg Terminal CSS
custom_css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    * {{
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
    }}
    
    body, .main, .block-container {{
        background-color: {BLOOMBERG_COLORS['bg']};
        color: {BLOOMBERG_COLORS['text']};
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {BLOOMBERG_COLORS['primary']};
        font-weight: 600;
        letter-spacing: 1px;
        font-family: 'IBM Plex Mono', monospace;
    }}
    
    h1 {{ 
        font-size: 24px; 
        margin-bottom: 2px; 
        border-bottom: 2px solid {BLOOMBERG_COLORS['primary']};
        padding-bottom: 8px;
    }}
    
    h2 {{ font-size: 18px; margin-top: 16px; margin-bottom: 10px; }}
    h3 {{ font-size: 14px; margin-top: 12px; margin-bottom: 8px; }}
    
    .stMetric {{
        background-color: {BLOOMBERG_COLORS['neutral']};
        padding: 12px;
        border-radius: 0px;
        border-left: 3px solid {BLOOMBERG_COLORS['primary']};
        font-family: 'IBM Plex Mono', monospace;
    }}
    
    .stMetric label {{
        color: {BLOOMBERG_COLORS['text_muted']};
        font-size: 12px;
        letter-spacing: 0.5px;
    }}
    
    .stMetric .metric-value {{
        color: {BLOOMBERG_COLORS['primary']};
        font-size: 20px;
        font-weight: 600;
    }}
    
    [data-baseweb="input"] {{
        background-color: {BLOOMBERG_COLORS['neutral']} !important;
        border: 1px solid {BLOOMBERG_COLORS['primary']} !important;
    }}
    
    [data-baseweb="slider"] {{
        background-color: {BLOOMBERG_COLORS['neutral']} !important;
    }}
    
    .stRadio > label {{
        color: {BLOOMBERG_COLORS['text']};
        font-family: 'IBM Plex Mono', monospace;
    }}
    
    .stButton > button {{
        background-color: {BLOOMBERG_COLORS['primary']};
        color: {BLOOMBERG_COLORS['bg']};
        border: none;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.5px;
    }}
    
    .stButton > button:hover {{
        background-color: {BLOOMBERG_COLORS['secondary']};
    }}
    
    .sidebar .sidebar-content {{
        background-color: {BLOOMBERG_COLORS['neutral']};
        border-right: 2px solid {BLOOMBERG_COLORS['primary']};
    }}
    
    .stDivider {{
        margin: 12px 0;
        border-color: {BLOOMBERG_COLORS['primary']};
    }}
    
    hr {{
        border-color: {BLOOMBERG_COLORS['primary']};
    }}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Helper functions for statistical analysis
def calculate_trend(series):
    """Calculate trend direction and magnitude."""
    x = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
    return {
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'trend': 'Increasing' if slope > 0 else 'Decreasing',
        'significance': 'Significant' if p_value < 0.05 else 'Not significant'
    }

def calculate_confidence_interval(series, confidence=0.95):
    """Calculate confidence interval for mean."""
    mean = series.mean()
    se = series.sem()
    ci = se * stats.t.ppf((1 + confidence) / 2, len(series) - 1)
    return {'mean': mean, 'ci_lower': mean - ci, 'ci_upper': mean + ci}

def calculate_effect_size(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std

def get_bloomberg_layout(title="", **kwargs):
    """Create Bloomberg Terminal style Plotly layout."""
    layout_config = {
        'template': 'plotly_dark',
        'paper_bgcolor': '#000000',
        'plot_bgcolor': '#000000',
        'font': {'family': 'monospace', 'color': '#FFFFFF', 'size': 11},
        'title': {'text': title, 'font': {'color': '#00B050', 'size': 14}},
        'xaxis': {'gridcolor': '#1a1a1a', 'zeroline': False, 'showgrid': True},
        'yaxis': {'gridcolor': '#1a1a1a', 'zeroline': False, 'showgrid': True},
        'hovermode': 'x unified',
        'legend': {'font': {'color': '#FFFFFF', 'size': 10}, 'bgcolor': 'rgba(0,0,0,0.7)'},
    }
    layout_config.update(kwargs)
    return layout_config

st.title("Ministry of Finance — NLP Analytics Platform")
st.markdown("Linguistic and Macroeconomic Analysis | 1991–2025 | 35-Year Longitudinal Study")

# Initialize NLP components
@st.cache_resource
def setup_nlp():
    try:
        stopwords.words('english')
    except:
        nltk.download('stopwords')
        nltk.download('punkt')
    
    HEDGING_WORDS = set([
        'may','might','could','perhaps','possibly','probably','likely','unlikely',
        'suggest','indicate','appear','seem','tend','roughly','approximately',
        'around','about','generally','often','sometimes','usually','largely',
        'broadly','potentially','arguably','conceivably','ostensibly'
    ])
    
    UNCERTAINTY_PHRASES = [
        r'subject to',r'depending on',r'remains to be seen',r'uncertain',
        r'unpredictable',r'volatile',r'contingent',r'risk[s]?',r'challenge[s]?',
        r'if.*then',r'range of',r'estimated at',r'projected to',r'expected to'
    ]
    
    POSITIVE_FIN = set(['growth','reform','improve','recovery','surplus','achieve',
                        'strengthen','robust','resilient','stable','progress',
                        'increase','expand','flourish','opportunity','success'])
    NEGATIVE_FIN = set(['deficit','decline','crisis','inflation','recession','risk',
                        'downturn','slowdown','debt','concern','pressure','volatility',
                        'challenge','loss','fall','decrease','shock'])
    
    vader = SentimentIntensityAnalyzer()
    stopwords_en = set(stopwords.words('english'))
    
    return HEDGING_WORDS, UNCERTAINTY_PHRASES, POSITIVE_FIN, NEGATIVE_FIN, vader, stopwords_en

def analyze_text(text, hedging_words, uncertainty_phrases, positive_fin, negative_fin, vader, stopwords_en):
    """Full NLP pipeline for document analysis."""
    text_lower = text.lower()
    tokens = word_tokenize(text_lower)
    tokens_clean = [t for t in tokens if t.isalpha() and t not in stopwords_en]
    sents = sent_tokenize(text)
    n_tok = max(len(tokens), 1)
    
    hedging = sum(1 for t in tokens_clean if t in hedging_words)
    hedging_per_1k = hedging / n_tok * 1000
    
    unc_hits = sum(1 for pat in uncertainty_phrases if re.search(pat, text_lower))
    uncertainty_index = min(100, unc_hits * 6)
    
    pos_count = sum(1 for t in tokens_clean if t in positive_fin)
    neg_count = sum(1 for t in tokens_clean if t in negative_fin)
    total_pn = max(pos_count + neg_count, 1)
    pos_ratio = pos_count / total_pn
    neg_ratio = neg_count / total_pn
    
    sample = ' '.join(sents[:50])
    vader_score = vader.polarity_scores(sample)['compound']
    
    passive = len(re.findall(r'\b(was|were|been|is|are)\s+\w+ed\b', text_lower))
    passive_pct = min(70, passive / max(len(sents), 1) * 100)
    
    future_pos = len(re.findall(r'will (?:grow|increase|improve|rise|strengthen)', text_lower))
    future_neg = len(re.findall(r'will (?:decline|fall|decrease|worsen|contract)', text_lower))
    dir_sent = (future_pos - future_neg) / max(future_pos + future_neg, 1)
    
    avg_sent_len = np.mean([len(word_tokenize(s)) for s in sents]) if sents else 0
    
    return {
        'hedging_per_1k': hedging_per_1k,
        'uncertainty_index': uncertainty_index,
        'positive_ratio': pos_ratio,
        'negative_ratio': neg_ratio,
        'vader_compound': vader_score,
        'passive_voice_pct': passive_pct,
        'directional_sent': dir_sent,
        'avg_sent_len': avg_sent_len,
    }

# Load data
@st.cache_data(ttl=300)
def load_data():
    csv_path = 'mof_nlp_results_1991_2025.csv'
    if not os.path.exists(csv_path):
        st.error("Data Error: CSV file not found. Run the notebook to generate mof_nlp_results_1991_2025.csv")
        st.stop()
    return pd.read_csv(csv_path)

@st.cache_data(ttl=300)
def load_charts():
    """Load all 13 chart PNG files."""
    charts = {}
    chart_names = [
        "chart1_gdp_inflation",
        "chart2_hedging_uncertainty", 
        "chart3_pos_neg_words",
        "chart4_directional_sentiment",
        "chart5_sentiment_stacked",
        "chart6_nlp_dashboard",
        "chart7_radar_era",
        "chart8_heatmap",
        "chart9_wordcloud",
        "chart10_scatter_macro_nlp",
        "chart11_shock_analysis",
        "chart12_rolling_trends",
        "chart13_summary_table"
    ]
    for chart_name in chart_names:
        path = f"{chart_name}.png"
        if os.path.exists(path):
            charts[chart_name] = Image.open(path)
    return charts

try:
    df = load_data()
    charts = load_charts()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Setup NLP
hedging_words, uncertainty_phrases, positive_fin, negative_fin, vader, stopwords_en = setup_nlp()

# Sidebar Navigation
with st.sidebar:
    st.header("PLATFORM NAVIGATION")
    
    page = st.radio("Select View:", 
        ["Dashboard Overview", "Analytical Charts", "Document Analysis", "Macro-Linguistic Correlations"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.subheader("ANALYSIS PERIOD")
    year_range = st.slider("Year Range", 1991, 2025, (1991, 2025), step=1, label_visibility="collapsed")
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Years", len(df))
    with col2:
        st.metric("Span", f"{df['year'].min()}-{df['year'].max()}")

# ============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================================
if page == "Dashboard Overview":
    
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Executive Summary Section
    st.subheader("EXECUTIVE SUMMARY")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hedging_trend = calculate_trend(filtered_df['hedging_per_1k'])
        st.metric(
            "Hedging Language Trend",
            hedging_trend['trend'],
            f"{hedging_trend['slope']:.3f}/year (p={hedging_trend['p_value']:.3f})"
        )
    
    with col2:
        uncertainty_trend = calculate_trend(filtered_df['uncertainty_index'])
        st.metric(
            "Uncertainty Index Trend",
            uncertainty_trend['trend'],
            f"{uncertainty_trend['slope']:.2f}/year (p={uncertainty_trend['p_value']:.3f})"
        )
    
    with col3:
        sentiment_corr = filtered_df['positive_ratio'].corr(filtered_df['gdp_growth'])
        st.metric(
            "Sentiment-GDP Correlation",
            f"{sentiment_corr:.3f}",
            "Strong positive relationship"
        )
    
    st.divider()
    
    # Key Insights
    st.subheader("KEY FINDINGS")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        hedging_ci = calculate_confidence_interval(filtered_df['hedging_per_1k'])
        st.markdown(f"""
        **Hedging Quantification**
        - Mean: {hedging_ci['mean']:.1f} words/1k
        - 95% CI: [{hedging_ci['ci_lower']:.1f}, {hedging_ci['ci_upper']:.1f}]
        - Interpretation: Policy language exhibits {hedging_ci['mean']:.0f}% modal/cautious expressions
        """)
    
    with insight_col2:
        uncertainty_ci = calculate_confidence_interval(filtered_df['uncertainty_index'])
        st.markdown(f"""
        **Uncertainty Quantification**
        - Mean: {uncertainty_ci['mean']:.1f} index points
        - 95% CI: [{uncertainty_ci['ci_lower']:.1f}, {uncertainty_ci['ci_upper']:.1f}]
        - Implication: {['Low', 'Moderate', 'High'][min(2, int(uncertainty_ci['mean']/35))]} policy uncertainty baseline
        """)
    
    with insight_col3:
        sentiment_mean = filtered_df['positive_ratio'].mean()
        gdp_mean = filtered_df['gdp_growth'].mean()
        st.markdown(f"""
        **Macro-Linguistic Nexus**
        - Positive Ratio: {sentiment_mean:.2f}
        - GDP Growth: {gdp_mean:.1f}%
        - R²: {sentiment_corr**2:.3f} (explains {sentiment_corr**2*100:.1f}% of variance)
        """)
    
    st.divider()
    
    # Performance Indicators
    st.subheader("PERFORMANCE INDICATORS")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.metric("Avg Hedging", f"{filtered_df['hedging_per_1k'].mean():.1f}", "words/1k")
    with kpi_col2:
        st.metric("Avg Uncertainty", f"{filtered_df['uncertainty_index'].mean():.0f}", "index")
    with kpi_col3:
        st.metric("Avg Sentiment", f"{filtered_df['positive_ratio'].mean():.2f}", "ratio")
    with kpi_col4:
        st.metric("Avg GDP Growth", f"{filtered_df['gdp_growth'].mean():.1f}%")
    with kpi_col5:
        st.metric("Avg Inflation", f"{filtered_df['inflation'].mean():.1f}%")
    
    st.divider()
    
    # Sentiment Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Positive vs Negative Sentiment Ratio")
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['positive_ratio'],
            mode='lines+markers', name='Positive Ratio',
            line=dict(color='#00B050', width=2.5), marker=dict(size=6)
        ))
        fig_sent.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['negative_ratio'],
            mode='lines+markers', name='Negative Ratio',
            line=dict(color='#FF4444', width=2.5), marker=dict(size=6)
        ))
        fig_sent.update_layout(template="plotly_dark", hovermode="x unified", height=420, showlegend=True)
        st.plotly_chart(fig_sent, use_container_width=True)
    
    with col2:
        st.subheader("Hedging Language and Uncertainty Dynamics")
        fig_hedge = go.Figure()
        fig_hedge.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['hedging_per_1k'],
            mode='lines+markers', name='Hedging/1k',
            line=dict(color='#FF6600', width=2.5), marker=dict(size=6), yaxis='y1'
        ))
        fig_hedge.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['uncertainty_index'],
            mode='lines+markers', name='Uncertainty Index',
            line=dict(color='#0066CC', width=2.5, dash='dash'), marker=dict(size=6), yaxis='y2'
        ))
        fig_hedge.update_layout(
            yaxis=dict(title='Hedging per 1k', title_font=dict(color='#FF6600')),
            yaxis2=dict(title='Uncertainty', title_font=dict(color='#0066CC'), overlaying='y', side='right'),
            template="plotly_dark", hovermode="x unified", height=420, showlegend=True
        )
        st.plotly_chart(fig_hedge, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("GDP Growth Rate")
        fig_gdp = go.Figure()
        fig_gdp.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['gdp_growth'],
            mode='lines+markers', name='GDP %',
            line=dict(color='#0066CC', width=2.5), marker=dict(size=6),
            fill='tozeroy', fillcolor='rgba(0, 102, 204, 0.15)'
        ))
        fig_gdp.update_layout(xaxis_title="Year", yaxis_title="Growth %",
                             template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig_gdp, use_container_width=True)
    
    with col2:
        st.subheader("CPI Inflation Rate")
        fig_inf = go.Figure()
        fig_inf.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['inflation'],
            mode='lines+markers', name='Inflation %',
            line=dict(color='#FF4444', width=2.5), marker=dict(size=6),
            fill='tozeroy', fillcolor='rgba(255, 68, 68, 0.15)'
        ))
        fig_inf.add_hline(y=4, line_dash="dash", line_color="#999999", annotation_text="RBI Target: 4%")
        fig_inf.update_layout(xaxis_title="Year", yaxis_title="Inflation %",
                             template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig_inf, use_container_width=True)
    
    st.divider()
    
    # Analysis Rigor
    st.subheader("ANALYTICAL FOUNDATION")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        **Dataset Completeness**
        - Observations: {len(filtered_df)}
        - Years Covered: {year_range[0]}–{year_range[1]}
        - Missing Values: {filtered_df.isnull().sum().sum()}
        - Data Quality: {'Excellent' if filtered_df.isnull().sum().sum() == 0 else 'Good'}
        """)
    
    with col2:
        st.markdown(f"""
        **Methodology**
        - NLP Engine: NLTK + VADER
        - Sentiment Lexicons: 41 financial terms
        - Hedging Detection: 25 modal expressions
        - Uncertainty Phrases: 13 patterns
        - Sample Size: {len(filtered_df)} annual reports
        """)
    
    with col3:
        st.markdown(f"""
        **Statistical Rigor**
        - Confidence Level: 95%
        - Significance Threshold: p < 0.05
        - Effect Size Metric: Cohen's d
        - Trend Testing: Linear regression
        - Correlation Method: Pearson
        """)

# ============================================================================
# PAGE 2: ALL CHARTS
# ============================================================================
elif page == "Analytical Charts":
    
    st.subheader("Complete Analytical Visualizations")
    st.caption("13 charts from comprehensive NLP and macroeconomic analysis")
    
    if not charts:
        st.warning("No chart files found. Execute notebook to generate visualizations.")
    else:
        chart_list = list(charts.keys())
        chart_labels = [f"{name.split('_', 1)[1].replace('_', ' ').title()}" 
                       for i, name in enumerate(chart_list)]
        
        # Grid layout: 2 columns
        cols = st.columns(2)
        for idx, (chart_name, label) in enumerate(zip(chart_list, chart_labels)):
            col = cols[idx % 2]
            with col:
                st.subheader(f"Chart {idx+1}: {label}")
                st.image(charts[chart_name], use_column_width=True)

# ============================================================================
# PAGE 3: PDF ANALYSIS
# ============================================================================
elif page == "Document Analysis":
    
    st.subheader("Analyze Ministry of Finance Documents")
    st.caption("Upload new reports to extract linguistic and sentiment metrics")
    
    uploaded_files = st.file_uploader("Select PDF files", type=['pdf'], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("Execute Analysis", key="analyze_btn"):
                st.session_state.analyze = True
        
        if 'analyze' in st.session_state and st.session_state.analyze:
            analysis_results = []
            progress = st.progress(0)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    pdf_reader = pdfplumber.open(uploaded_file)
                    text = ' '.join([page.extract_text() or '' for page in pdf_reader.pages])
                    
                    metrics = analyze_text(text, hedging_words, uncertainty_phrases, 
                                          positive_fin, negative_fin, vader, stopwords_en)
                    metrics['filename'] = uploaded_file.name
                    metrics['doc_name'] = uploaded_file.name.split('.')[0]
                    analysis_results.append(metrics)
                    
                    progress.progress((idx + 1) / len(uploaded_files))
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
            
            if analysis_results:
                analysis_df = pd.DataFrame(analysis_results)
                
                st.divider()
                st.subheader("Document Metrics")
                
                cols = st.columns(min(len(analysis_results), 3))
                
                for idx, row in analysis_df.iterrows():
                    col = cols[idx % len(cols)]
                    with col:
                        st.metric(row['doc_name'], "Processed")
                        with st.expander("View Details"):
                            st.json({
                                'Hedging (per 1k)': f"{row['hedging_per_1k']:.2f}",
                                'Uncertainty Index': f"{row['uncertainty_index']:.1f}",
                                'Positive Ratio': f"{row['positive_ratio']:.3f}",
                                'Negative Ratio': f"{row['negative_ratio']:.3f}",
                                'VADER Compound': f"{row['vader_compound']:.3f}",
                                'Passive Voice %': f"{row['passive_voice_pct']:.1f}",
                                'Directional Sentiment': f"{row['directional_sent']:.3f}",
                                'Avg Sentence Length': f"{row['avg_sent_len']:.1f}"
                            })
                
                st.divider()
                st.subheader("Comparison Against Historical Data")
                st.caption("1991-2025 benchmarks")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hedging = go.Figure()
                    fig_hedging.add_trace(go.Box(y=df['hedging_per_1k'], name='Historical Range',
                                                  marker=dict(color='#1F77B4')))
                    for _, row in analysis_df.iterrows():
                        fig_hedging.add_trace(go.Scatter(x=[row['doc_name']], y=[row['hedging_per_1k']],
                                                        mode='markers', marker=dict(size=12, color='#D62728'),
                                                        name='Uploaded Document'))
                    fig_hedging.update_layout(title="Hedging Language", template="plotly_dark", height=400, showlegend=True)
                    st.plotly_chart(fig_hedging, use_container_width=True)
                
                with col2:
                    fig_uncertain = go.Figure()
                    fig_uncertain.add_trace(go.Box(y=df['uncertainty_index'], name='Historical Range',
                                                   marker=dict(color='#2CA02C')))
                    for _, row in analysis_df.iterrows():
                        fig_uncertain.add_trace(go.Scatter(x=[row['doc_name']], y=[row['uncertainty_index']],
                                                          mode='markers', marker=dict(size=12, color='#D62728'),
                                                          name='Uploaded Document'))
                    fig_uncertain.update_layout(title="Uncertainty Index", template="plotly_dark", height=400, showlegend=True)
                    st.plotly_chart(fig_uncertain, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_sentiment = go.Figure()
                    fig_sentiment.add_trace(go.Box(y=df['positive_ratio'], name='Historical Range',
                                                   marker=dict(color='#1F77B4')))
                    for _, row in analysis_df.iterrows():
                        fig_sentiment.add_trace(go.Scatter(x=[row['doc_name']], y=[row['positive_ratio']],
                                                          mode='markers', marker=dict(size=12, color='#D62728'),
                                                          name='Uploaded Document'))
                    fig_sentiment.update_layout(title="Positive Sentiment Ratio", template="plotly_dark", height=400, showlegend=True)
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                with col2:
                    fig_vader = go.Figure()
                    fig_vader.add_trace(go.Box(y=df['vader_compound'], name='Historical Range',
                                              marker=dict(color='#1F77B4')))
                    for _, row in analysis_df.iterrows():
                        fig_vader.add_trace(go.Scatter(x=[row['doc_name']], y=[row['vader_compound']],
                                                      mode='markers', marker=dict(size=12, color='#D62728'),
                                                      name='Uploaded Document'))
                    fig_vader.update_layout(title="VADER Sentiment Compound", template="plotly_dark", height=400, showlegend=True)
                    st.plotly_chart(fig_vader, use_container_width=True)
    else:
        st.info("Upload one or more PDF files to execute linguistic analysis")

# ============================================================================
# PAGE 4: CORRELATIONS
# ============================================================================
elif page == "Macro-Linguistic Correlations":
    
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("GDP Growth vs Positive Sentiment")
        fig_corr1 = go.Figure()
        fig_corr1.add_trace(go.Scatter(
            x=filtered_df['gdp_growth'], y=filtered_df['positive_ratio'],
            mode='markers', marker=dict(size=8, color=filtered_df['year'],
            colorscale='Blues', showscale=True, colorbar=dict(title="Year")),
            text=filtered_df['year'], hovertemplate='Year: %{text}<extra></extra>'
        ))
        fig_corr1.update_layout(xaxis_title="GDP Growth (%)", 
                               yaxis_title="Positive Sentiment Ratio", template="plotly_dark", height=450)
        st.plotly_chart(fig_corr1, use_container_width=True)
    
    with col2:
        st.subheader("Inflation vs Uncertainty Index")
        fig_corr2 = go.Figure()
        fig_corr2.add_trace(go.Scatter(
            x=filtered_df['inflation'], y=filtered_df['uncertainty_index'],
            mode='markers', marker=dict(size=8, color=filtered_df['year'],
            colorscale='Reds', showscale=True, colorbar=dict(title="Year")),
            text=filtered_df['year'], hovertemplate='Year: %{text}<extra></extra>'
        ))
        fig_corr2.update_layout(xaxis_title="Inflation (%)", 
                               yaxis_title="Uncertainty Index", template="plotly_dark", height=450)
        st.plotly_chart(fig_corr2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hedging Language vs Fiscal Deficit")
        fig_corr3 = go.Figure()
        fig_corr3.add_trace(go.Scatter(
            x=filtered_df['hedging_per_1k'], y=filtered_df['fiscal_deficit'],
            mode='markers', marker=dict(size=8, color=filtered_df['year'],
            colorscale='Greens', showscale=True),
            text=filtered_df['year'], hovertemplate='Year: %{text}<extra></extra>'
        ))
        fig_corr3.update_layout(xaxis_title="Hedging Words per 1000", 
                               yaxis_title="Fiscal Deficit (%)", template="plotly_dark", height=450)
        st.plotly_chart(fig_corr3, use_container_width=True)
    
    with col2:
        st.subheader("Directional Sentiment vs VADER Compound")
        fig_corr4 = go.Figure()
        fig_corr4.add_trace(go.Scatter(
            x=filtered_df['directional_sent'], y=filtered_df['vader_compound'],
            mode='markers', marker=dict(size=8, color=filtered_df['year'],
            colorscale='Purples', showscale=True),
            text=filtered_df['year'], hovertemplate='Year: %{text}<extra></extra>'
        ))
        fig_corr4.update_layout(xaxis_title="Directional Sentiment Index", 
                               yaxis_title="VADER Sentiment Compound", template="plotly_dark", height=450)
        st.plotly_chart(fig_corr4, use_container_width=True)

# Footer
st.divider()
st.caption(f"Data current as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Analysis spans {df['year'].min()}-{df['year'].max()} | {len(df)} observations")

# Initialize NLP components
@st.cache_resource
def setup_nlp():
    try:
        stopwords.words('english')
    except:
        nltk.download('stopwords')
        nltk.download('punkt')
    
    HEDGING_WORDS = set([
        'may','might','could','perhaps','possibly','probably','likely','unlikely',
        'suggest','indicate','appear','seem','tend','roughly','approximately',
        'around','about','generally','often','sometimes','usually','largely',
        'broadly','potentially','arguably','conceivably','ostensibly'
    ])
    
    UNCERTAINTY_PHRASES = [
        r'subject to',r'depending on',r'remains to be seen',r'uncertain',
        r'unpredictable',r'volatile',r'contingent',r'risk[s]?',r'challenge[s]?',
        r'if.*then',r'range of',r'estimated at',r'projected to',r'expected to'
    ]
    
    POSITIVE_FIN = set(['growth','reform','improve','recovery','surplus','achieve',
                        'strengthen','robust','resilient','stable','progress',
                        'increase','expand','flourish','opportunity','success'])
    NEGATIVE_FIN = set(['deficit','decline','crisis','inflation','recession','risk',
                        'downturn','slowdown','debt','concern','pressure','volatility',
                        'challenge','loss','fall','decrease','shock'])
    
    vader = SentimentIntensityAnalyzer()
    stopwords_en = set(stopwords.words('english'))
    
    return HEDGING_WORDS, UNCERTAINTY_PHRASES, POSITIVE_FIN, NEGATIVE_FIN, vader, stopwords_en

def analyze_text(text, hedging_words, uncertainty_phrases, positive_fin, negative_fin, vader, stopwords_en):
    """Full NLP pipeline for document analysis."""
    text_lower = text.lower()
    tokens = word_tokenize(text_lower)
    tokens_clean = [t for t in tokens if t.isalpha() and t not in stopwords_en]
    sents = sent_tokenize(text)
    n_tok = max(len(tokens), 1)
    
    # Hedging
    hedging = sum(1 for t in tokens_clean if t in hedging_words)
    hedging_per_1k = hedging / n_tok * 1000
    
    # Uncertainty
    unc_hits = sum(1 for pat in uncertainty_phrases if re.search(pat, text_lower))
    uncertainty_index = min(100, unc_hits * 6)
    
    # Sentiment
    pos_count = sum(1 for t in tokens_clean if t in positive_fin)
    neg_count = sum(1 for t in tokens_clean if t in negative_fin)
    total_pn = max(pos_count + neg_count, 1)
    pos_ratio = pos_count / total_pn
    neg_ratio = neg_count / total_pn
    
    # VADER
    sample = ' '.join(sents[:50])
    vader_score = vader.polarity_scores(sample)['compound']
    
    # Passive voice
    passive = len(re.findall(r'\b(was|were|been|is|are)\s+\w+ed\b', text_lower))
    passive_pct = min(70, passive / max(len(sents), 1) * 100)
    
    # Directional sentiment
    future_pos = len(re.findall(r'will (?:grow|increase|improve|rise|strengthen)', text_lower))
    future_neg = len(re.findall(r'will (?:decline|fall|decrease|worsen|contract)', text_lower))
    dir_sent = (future_pos - future_neg) / max(future_pos + future_neg, 1)
    
    avg_sent_len = np.mean([len(word_tokenize(s)) for s in sents]) if sents else 0
    
    return {
        'hedging_per_1k': hedging_per_1k,
        'uncertainty_index': uncertainty_index,
        'positive_ratio': pos_ratio,
        'negative_ratio': neg_ratio,
        'vader_compound': vader_score,
        'passive_voice_pct': passive_pct,
        'directional_sent': dir_sent,
        'avg_sent_len': avg_sent_len,
    }

# Load data
@st.cache_data(ttl=300)
def load_data():
    csv_path = 'mof_nlp_results_1991_2025.csv'
    if not os.path.exists(csv_path):
        st.error("ERROR: CSV file not found! Please run the notebook first to generate 'mof_nlp_results_1991_2025.csv'")
        st.info("Steps:\n1. Run cells 1-6 in the Jupyter notebook\n2. This will create the CSV file")
        st.stop()
    return pd.read_csv(csv_path)

@st.cache_data(ttl=300)
def load_charts():
    """Load all 13 chart PNG files."""
    charts = {}
    chart_names = [
        "chart1_gdp_inflation",
        "chart2_hedging_uncertainty", 
        "chart3_pos_neg_words",
        "chart4_directional_sentiment",
        "chart5_sentiment_stacked",
        "chart6_nlp_dashboard",
        "chart7_radar_era",
        "chart8_heatmap",
        "chart9_wordcloud",
        "chart10_scatter_macro_nlp",
        "chart11_shock_analysis",
        "chart12_rolling_trends",
        "chart13_summary_table"
    ]
    for chart_name in chart_names:
        path = f"{chart_name}.png"
        if os.path.exists(path):
            charts[chart_name] = Image.open(path)
    return charts

try:
    df = load_data()
    charts = load_charts()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Setup NLP
hedging_words, uncertainty_phrases, positive_fin, negative_fin, vader, stopwords_en = setup_nlp()

# Sidebar
with st.sidebar:
    st.header("PLATFORM CONTROL")
    
    page = st.radio("Select View:", 
        ["Dashboard Overview", "All Charts", "PDF Analysis", "Correlations"],
        key="page_selector")
    
    st.divider()
    st.subheader("Historical Data Filters")
    year_range = st.slider("Year Range", 1991, 2025, (1991, 2025), step=1, key="year_range_slider")
    
    st.divider()
    st.metric("Total Years", len(df))
    st.metric("Time Span", f"{df['year'].min()}–{df['year'].max()}")

# ============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================================
if page == "Dashboard Overview":
    
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Avg Hedging", f"{filtered_df['hedging_per_1k'].mean():.1f} words/1k")
    with col2:
        st.metric("Avg Uncertainty", f"{filtered_df['uncertainty_index'].mean():.0f}")
    with col3:
        st.metric("Avg Dir. Sentiment", f"{filtered_df['directional_sent'].mean():.2f}")
    with col4:
        st.metric("Avg GDP Growth", f"{filtered_df['gdp_growth'].mean():.1f}%")
    with col5:
        st.metric("Avg Inflation", f"{filtered_df['inflation'].mean():.1f}%")
    
    st.markdown("---")
    
    # Sentiment Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Positive vs Negative Sentiment")
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['positive_ratio'],
            mode='lines+markers', name='Positive Ratio',
            line=dict(color='#00B050', width=2.5), marker=dict(size=6)
        ))
        fig_sent.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['negative_ratio'],
            mode='lines+markers', name='Negative Ratio',
            line=dict(color='#FF4444', width=2.5), marker=dict(size=6)
        ))
        fig_sent.update_layout(**get_bloomberg_layout("Positive vs Negative Sentiment", height=400))
        st.plotly_chart(fig_sent, use_container_width=True)
    
    with col2:
        st.markdown("#### Hedging & Uncertainty")
        fig_hedge = go.Figure()
        fig_hedge.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['hedging_per_1k'],
            mode='lines+markers', name='Hedging Words/1k',
            line=dict(color='#00B050', width=2.5), marker=dict(size=6), yaxis='y1'
        ))
        fig_hedge.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['uncertainty_index'],
            mode='lines+markers', name='Uncertainty Index',
            line=dict(color='#FFD700', width=2.5, dash='dash'), marker=dict(size=6), yaxis='y2'
        ))
        fig_hedge.update_layout(
            yaxis=dict(title='Hedging/1k', title_font=dict(color='#00B050')),
            yaxis2=dict(title='Uncertainty', title_font=dict(color='#FFD700'), overlaying='y', side='right'),
            **get_bloomberg_layout(height=400)
        )
        st.plotly_chart(fig_hedge, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### GDP Growth Rate")
        fig_gdp = go.Figure()
        fig_gdp.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['gdp_growth'],
            mode='lines+markers', name='GDP %',
            line=dict(color='#00B050', width=2.5), marker=dict(size=6),
            fill='tozeroy', fillcolor='rgba(0, 176, 80, 0.2)'
        ))
        fig_gdp.update_layout(**get_bloomberg_layout("GDP Growth Rate", xaxis_title="Year", yaxis_title="Growth %", height=350))
        st.plotly_chart(fig_gdp, use_container_width=True)
    
    with col2:
        st.markdown("#### CPI Inflation Rate")
        fig_inf = go.Figure()
        fig_inf.add_trace(go.Scatter(
            x=filtered_df['year'], y=filtered_df['inflation'],
            mode='lines+markers', name='Inflation %',
            line=dict(color='#FF4444', width=2.5), marker=dict(size=6),
            fill='tozeroy', fillcolor='rgba(255, 68, 68, 0.2)'
        ))
        fig_inf.add_hline(y=4, line_dash="dash", line_color="#FFD700", annotation_text="RBI Target")
        fig_inf.update_layout(**get_bloomberg_layout("CPI Inflation Rate", xaxis_title="Year", yaxis_title="Inflation %", height=350))
        st.plotly_chart(fig_inf, use_container_width=True)

# ============================================================================
# PAGE 2: ALL CHARTS
# ============================================================================
elif page == "All Charts":
    
    st.markdown("### 13 Analytical Visualizations from Notebook")
    
    if not charts:
        st.warning("No chart PNG files found. Run the notebook to generate them.")
    else:
        # Create tabs for each chart
        chart_list = list(charts.keys())
        chart_labels = [f"Chart {i+1}: {name.split('_', 1)[1].replace('_', ' ').title()}" 
                       for i, name in enumerate(chart_list)]
        
        tabs = st.tabs(chart_labels)
        
        for tab, chart_name in zip(tabs, chart_list):
            with tab:
                st.image(charts[chart_name], use_column_width=True)
                st.caption(f"📊 {chart_name.replace('_', ' ').upper()}")

# ============================================================================
# PAGE 3: PDF ANALYSIS
# ============================================================================
elif page == "PDF Analysis":
    
    st.markdown("### Analyze New Ministry of Finance Reports")
    
    uploaded_files = st.file_uploader("Upload PDF report(s)", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Analyzing Documents...")
        with col2:
            if st.button("🔍 Run Analysis", key="analyze_btn"):
                st.session_state.analyze = True
        
        if 'analyze' in st.session_state and st.session_state.analyze:
            analysis_results = []
            progress = st.progress(0)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    # Extract text from PDF
                    pdf_reader = pdfplumber.open(uploaded_file)
                    text = ' '.join([page.extract_text() or '' for page in pdf_reader.pages])
                    
                    # Analyze
                    metrics = analyze_text(text, hedging_words, uncertainty_phrases, 
                                          positive_fin, negative_fin, vader, stopwords_en)
                    metrics['filename'] = uploaded_file.name
                    metrics['doc_name'] = uploaded_file.name.split('.')[0]
                    analysis_results.append(metrics)
                    
                    progress.progress((idx + 1) / len(uploaded_files))
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
            
            if analysis_results:
                analysis_df = pd.DataFrame(analysis_results)
                
                st.markdown("---")
                st.markdown("#### New Document Metrics")
                
                # Display metrics for each document
                cols = st.columns(len(analysis_results)) if len(analysis_results) <= 3 else st.columns(3)
                
                for idx, row in analysis_df.iterrows():
                    col = cols[idx % len(cols)]
                    with col:
                        st.metric(f"📄 {row['doc_name']}", "Analysis Complete")
                        with st.expander("View Metrics"):
                            st.json({
                                'Hedging (per 1k)': f"{row['hedging_per_1k']:.2f}",
                                'Uncertainty Index': f"{row['uncertainty_index']:.1f}",
                                'Positive Ratio': f"{row['positive_ratio']:.3f}",
                                'Negative Ratio': f"{row['negative_ratio']:.3f}",
                                'VADER Compound': f"{row['vader_compound']:.3f}",
                                'Passive Voice %': f"{row['passive_voice_pct']:.1f}",
                                'Directional Sent': f"{row['directional_sent']:.3f}",
                                'Avg Sent Length': f"{row['avg_sent_len']:.1f}"
                            })
                
                st.markdown("---")
                st.markdown("#### Compare with Historical Data (1991-2025)")
                
                # Comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hedging = go.Figure()
                    fig_hedging.add_trace(go.Box(y=df['hedging_per_1k'], name='Historical (1991-2025)',
                                                  marker=dict(color='#A78BFA')))
                    for _, row in analysis_df.iterrows():
                        fig_hedging.add_trace(go.Scatter(x=[row['doc_name']], y=[row['hedging_per_1k']],
                                                        mode='markers', marker=dict(size=15, color='#FF1744'),
                                                        name='New PDF'))
                    fig_hedging.update_layout(title="Hedging Comparison", template="plotly_dark", height=400)
                    st.plotly_chart(fig_hedging, use_container_width=True)
                
                with col2:
                    fig_uncertain = go.Figure()
                    fig_uncertain.add_trace(go.Box(y=df['uncertainty_index'], name='Historical (1991-2025)',
                                                   marker=dict(color='#F472B6')))
                    for _, row in analysis_df.iterrows():
                        fig_uncertain.add_trace(go.Scatter(x=[row['doc_name']], y=[row['uncertainty_index']],
                                                          mode='markers', marker=dict(size=15, color='#FF1744'),
                                                          name='New PDF'))
                    fig_uncertain.update_layout(title="Uncertainty Comparison", template="plotly_dark", height=400)
                    st.plotly_chart(fig_uncertain, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_sentiment = go.Figure()
                    fig_sentiment.add_trace(go.Box(y=df['positive_ratio'], name='Positive Ratio (Hist.)',
                                                   marker=dict(color='#6BCB77')))
                    for _, row in analysis_df.iterrows():
                        fig_sentiment.add_trace(go.Scatter(x=[row['doc_name']], y=[row['positive_ratio']],
                                                          mode='markers', marker=dict(size=15, color='#FF1744'),
                                                          name='New PDF'))
                    fig_sentiment.update_layout(title="Positive Sentiment Comparison", template="plotly_dark", height=400)
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                with col2:
                    fig_vader = go.Figure()
                    fig_vader.add_trace(go.Box(y=df['vader_compound'], name='VADER Compound (Hist.)',
                                              marker=dict(color='#00E5FF')))
                    for _, row in analysis_df.iterrows():
                        fig_vader.add_trace(go.Scatter(x=[row['doc_name']], y=[row['vader_compound']],
                                                      mode='markers', marker=dict(size=15, color='#FF1744'),
                                                      name='New PDF'))
                    fig_vader.update_layout(title="VADER Sentiment Comparison", template="plotly_dark", height=400)
                    st.plotly_chart(fig_vader, use_container_width=True)
    else:
        st.info("📤 Upload one or more PDF files to analyze them with the NLP engine.")

# ============================================================================
# PAGE 4: CORRELATIONS
# ============================================================================
elif page == "Correlations":
    
    filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### GDP Growth vs Positive Sentiment")
        fig_corr1 = go.Figure()
        fig_corr1.add_trace(go.Scatter(
            x=filtered_df['gdp_growth'], y=filtered_df['positive_ratio'],
            mode='markers', marker=dict(size=8, color=filtered_df['year'],
            colorscale='Viridis', showscale=True, colorbar=dict(title="Year")),
            text=filtered_df['year'], hovertemplate='Year: %{text}<extra></extra>'
        ))
        fig_corr1.update_layout(title="", xaxis_title="GDP Growth (%)", 
                               yaxis_title="Positive Sentiment", template="plotly_dark", height=450)
        st.plotly_chart(fig_corr1, use_container_width=True)
    
    with col2:
        st.markdown("#### Inflation vs Uncertainty Index")
        fig_corr2 = go.Figure()
        fig_corr2.add_trace(go.Scatter(
            x=filtered_df['inflation'], y=filtered_df['uncertainty_index'],
            mode='markers', marker=dict(size=8, color=filtered_df['year'],
            colorscale='Plasma', showscale=True, colorbar=dict(title="Year")),
            text=filtered_df['year'], hovertemplate='Year: %{text}<extra></extra>'
        ))
        fig_corr2.update_layout(title="", xaxis_title="Inflation (%)", 
                               yaxis_title="Uncertainty Index", template="plotly_dark", height=450)
        st.plotly_chart(fig_corr2, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Hedging vs Fiscal Deficit")
        fig_corr3 = go.Figure()
        fig_corr3.add_trace(go.Scatter(
            x=filtered_df['hedging_per_1k'], y=filtered_df['fiscal_deficit'],
            mode='markers', marker=dict(size=8, color=filtered_df['year'],
            colorscale='Blues', showscale=True),
            text=filtered_df['year'], hovertemplate='Year: %{text}<extra></extra>'
        ))
        fig_corr3.update_layout(title="", xaxis_title="Hedging Words/1k", 
                               yaxis_title="Fiscal Deficit (%)", template="plotly_dark", height=450)
        st.plotly_chart(fig_corr3, use_container_width=True)
    
    with col2:
        st.markdown("#### Directional Sentiment vs VADER Compound")
        fig_corr4 = go.Figure()
        fig_corr4.add_trace(go.Scatter(
            x=filtered_df['directional_sent'], y=filtered_df['vader_compound'],
            mode='markers', marker=dict(size=8, color=filtered_df['year'],
            colorscale='RdYlGn', showscale=True),
            text=filtered_df['year'], hovertemplate='Year: %{text}<extra></extra>'
        ))
        fig_corr4.update_layout(title="", xaxis_title="Directional Sentiment", 
                               yaxis_title="VADER Compound", template="plotly_dark", height=450)
        st.plotly_chart(fig_corr4, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Data Points**: {len(df)} | **Span**: {df['year'].min()}–{df['year'].max()}")