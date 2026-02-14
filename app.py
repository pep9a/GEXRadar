import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 1. SETUP & BRANDING
st.set_page_config(page_title="GEXRADAR // QUANT TERMINAL", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
        .block-container { padding-top: 0.5rem !important; padding-bottom: 0rem !important; }
        header {visibility: hidden;} 
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0E1117; color: #E0E0E0; }
        
        /* STICKY HEADER */
        .sticky-header {
            position: sticky;
            top: 0;
            background-color: #0E1117;
            z-index: 1000;
            padding: 10px 0px;
            border-bottom: 1px solid #30363D;
            margin-bottom: 20px;
        }

        .logo-text { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 700; color: #00C805; letter-spacing: -1px; }
        .logo-underscore { color: #FFFFFF; }
        
        /* SIDEBAR OG STYLE */
        .sidebar-label { font-size: 10px; color: #666; letter-spacing: 1.5px; font-weight: 800; text-transform: uppercase; margin-top: 20px; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 3px; }
        .data-block { background: rgba(255, 255, 255, 0.03); border: 1px solid #30363D; padding: 12px; border-radius: 4px; margin-bottom: 8px; }
        .data-label { font-size: 10px; color: #8B949E; margin-bottom: 2px; text-transform: uppercase; }
        .data-value { font-family: 'JetBrains Mono'; font-size: 16px; font-weight: 700; color: #E6EDF3; }
        
        .regime-container { padding: 20px; border-radius: 4px; border: 1px solid #30363D; background: #161B22; margin-bottom: 25px; }
        
        /* FANCY MODE BUTTONS */
        .stButton > button {
            width: 100%;
            border-radius: 4px;
            border: 1px solid #30363D;
            background: #161B22;
            color: #8B949E;
            font-family: 'JetBrains Mono';
            font-size: 12px;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            border-color: #00C805;
            color: #00C805;
            background: rgba(0, 200, 5, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA ENGINE
strikes = np.arange(580, 605, 0.5)
spot_price = 592.40
gamma_flip_level = 591.0
momentum_wall = 595.0
max_pain = 590.0
vol_trigger = 588.5

# Time-Series Simulation
times = pd.date_range(start='9:30', periods=30, freq='10min')
price_walk = 592 + np.cumsum(np.random.normal(0, 0.4, 30))
t_series = pd.DataFrame({
    'Time': times, 'Price': price_walk,
    'GEX': np.random.uniform(-400, 900, 30),
    'DEX': np.random.uniform(20, 150, 30),
    'CNV': np.random.uniform(-150, 350, 30)
})

data = []
for s in strikes:
    is_major = 3.5 if s % 5 == 0 else (1.8 if s % 2.5 == 0 else 0.6)
    call_g = np.exp(-(abs(s - 595)**2)/12) * is_major
    put_g = np.exp(-(abs(s - 588)**2)/12) * is_major
    data.append({
        'strike': s, 'call_gex': call_g * 6000, 'put_gex': -put_g * 6000,
        'vanna': (call_g + put_g) * 0.4, 'oi': int((call_g + put_g) * 60000),
        'vol_call': np.random.randint(1500, 9000) * call_g, 'vol_put': np.random.randint(1500, 9000) * put_g,
        'vega': (call_g + put_g) * 0.25, 'charm': (call_g - put_g) * 0.12,
        'iv': 0.15 + (abs(s - 592)**2 * 0.0008) # Smile curvature
    })
df = pd.DataFrame(data)

flow_ratio = df['vol_call'].sum() / (df['vol_call'].sum() + df['vol_put'].sum())
is_long_gamma = spot_price > gamma_flip_level
regime_color = "#00C805" if is_long_gamma else "#FF3B3B"

# 3. STICKY HEADER
with st.container():
    st.markdown('<div class="sticky-header"><div class="logo-text">GEXRADAR<span class="logo-underscore">_</span></div></div>', unsafe_allow_html=True)
    if 'current_page' not in st.session_state: st.session_state.current_page = "DASHBOARD"
    col_nav1, col_nav2, col_nav3, _ = st.columns([1.2, 1.2, 1.2, 7])
    with col_nav1:
        if st.button("DASHBOARD"): st.session_state.current_page = "DASHBOARD"
    with col_nav2:
        if st.button("OPERATIONS"): st.session_state.current_page = "OPERATIONS"
    with col_nav3:
        if st.button("ABOUT"): st.session_state.current_page = "ABOUT"
    st.markdown("---")

# --- DASHBOARD PAGE ---
if st.session_state.current_page == "DASHBOARD":
    # SIDEBAR: OG FULL PACKED
    st.sidebar.markdown("<p class='sidebar-label'>Structural Context</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
        <div class="data-block"><div class="data-label">Flow Ratio</div><div class="data-value" style="color:{regime_color}">{flow_ratio:.2f}</div></div>
        <div class="data-block"><div class="data-label">Momentum Wall</div><div class="data-value">${momentum_wall}</div></div>
        <div class="data-block"><div class="data-label">Max Pain</div><div class="data-value">${max_pain}</div></div>
        <div class="data-block"><div class="data-label">Zero Gamma</div><div class="data-value">${gamma_flip_level}</div></div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<p class='sidebar-label'>Risk & Volatility</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
        <div class="data-block"><div class="data-label">Vol Trigger</div><div class="data-value" style="color:#FF3B3B">${vol_trigger}</div></div>
        <div class="data-block"><div class="data-label">Vanna Exposure</div><div class="data-value">${df['vanna'].sum():.2f}M</div></div>
        <div class="data-block"><div class="data-label">Total OI</div><div class="data-value">{df['oi'].sum():,}</div></div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<p class='sidebar-label'>Topography Scan</p>", unsafe_allow_html=True)
    topo = df[(df['strike'] > spot_price - 8) & (df['strike'] < spot_price + 8)].sort_values('strike')
    z_matrix = np.outer(np.linspace(1, 0.1, 10), (np.abs(topo['call_gex']) + np.abs(topo['put_gex'])).values)
    fig_3d_sidebar = go.Figure(data=[go.Surface(z=z_matrix, x=topo['strike'].values, colorscale='Viridis', showscale=False)])
    fig_3d_sidebar.update_layout(height=180, margin=dict(l=0,r=0,b=0,t=0), template="plotly_dark")
    st.sidebar.plotly_chart(fig_3d_sidebar, use_container_width=True)

    # MAIN CONTENT
    st.markdown(f"""<div class="regime-container"><div style="font-size: 11px; color: #808495; text-transform: uppercase;">Market Regime</div>
        <div style="font-family: 'JetBrains Mono'; font-size: 24px; font-weight: 700; color: {regime_color};">{"STABLE / LONG GAMMA" if is_long_gamma else "VOLATILE / SHORT GAMMA"}</div></div>""", unsafe_allow_html=True)

    # FANCY RADAR SELECTION
    if 'radar_mode' not in st.session_state: st.session_state.radar_mode = "GEX"
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1: 
        if st.button("OPEN INTEREST"): st.session_state.radar_mode = "GEX"
    with col_m2: 
        if st.button("VOLUME FLOW"): st.session_state.radar_mode = "VOL"
    with col_m3: 
        if st.button("HEATMAP"): st.session_state.radar_mode = "HEAT"
    with col_m4: 
        if st.button("VOL SURFACE"): st.session_state.radar_mode = "SURF"

    plot_df = df[(df['strike'] > spot_price - 12) & (df['strike'] < spot_price + 12)]
    
    # ADVANCED MODE LOGIC
    if st.session_state.radar_mode == "SURF":
        # Advanced 3D Surface with Smile Interpolation
        days = np.array([1, 7, 30, 60, 90])
        z_vol = np.array([plot_df['iv'].values * (1 + 0.05 * np.log(d)) for d in days])
        fig_main = go.Figure(data=[go.Surface(z=z_vol, x=plot_df['strike'], y=days, 
                                            colorscale='Thermal', 
                                            contours_z=dict(show=True, usecolormap=True, project_z=True, highlightcolor="limegreen"))])
        fig_main.update_layout(scene=dict(xaxis_title='Strike', yaxis_title='DTE', zaxis_title='IV'), height=700)
    
    elif st.session_state.radar_mode == "HEAT":
        # Professional Density Heatmap
        heat_data = np.random.randn(len(plot_df), 20) # Simulated flow density
        fig_main = go.Figure(data=go.Heatmap(z=heat_data, x=np.arange(20), y=plot_df['strike'], 
                                            colorscale='Magma', hoverongaps=False))
        fig_main.update_layout(height=700, yaxis_title="Strike Price", xaxis_title="Time Interval (Relative)")

    else:
        fig_main = go.Figure()
        if st.session_state.radar_mode == "GEX":
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['call_gex'], orientation='h', marker_color='#00C805', name="Call GEX"))
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['put_gex'], orientation='h', marker_color='#FF3B3B', name="Put GEX"))
        else:
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['vol_call'], orientation='h', marker_color='#00C805', name="Call Vol"))
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=-plot_df['vol_put'], orientation='h', marker_color='#FF3B3B', name="Put Vol"))
        
        # Annotations (Price Levels)
        fig_main.add_hline(y=gamma_flip_level, line_dash="dot", line_color="#808495", annotation_text="FLIP")
        fig_main.add_hline(y=spot_price, line_color="#FFFFFF", line_width=2, annotation_text="SPOT")
        fig_main.add_hline(y=momentum_wall, line_dash="dash", line_color="#00FFFF", annotation_text="MOM WALL")
        fig_main.add_hline(y=max_pain, line_dash="dot", line_color="#FFD700", annotation_text="MAX PAIN")
        fig_main.add_hline(y=vol_trigger, line_dash="dashdot", line_color="#FF3B3B", annotation_text="VOL TRIGGER")
        fig_main.update_layout(height=700, bargap=0.1, yaxis_title="Strike Price")

    fig_main.update_layout(template="plotly_dark", margin=dict(t=30))
    st.plotly_chart(fig_main, use_container_width=True)

    # TRIPLE CHARTS & GREEK MATRIX (PRESERVED)
    st.markdown("### Institutional Time-Series ($MM)")
    # ...[Triple chart code preserved]...
    def create_ts_chart(y_col, color, y_range, title):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=t_series['Time'], y=t_series[y_col], name=title, line=dict(color=color, width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=t_series['Time'], y=t_series['Price'], name="Price", line=dict(color='white', width=1, dash='dot'), opacity=0.4), secondary_y=True)
        fig.update_yaxes(range=y_range, secondary_y=False, gridcolor='#222')
        fig.update_layout(height=280, template="plotly_dark", margin=dict(t=30, b=20), hovermode='x unified', showlegend=False, title=title)
        return fig

    st.plotly_chart(create_ts_chart('GEX', '#00C805', [-500, 1000], "GEX $MM"), use_container_width=True)
    st.plotly_chart(create_ts_chart('DEX', '#00FFFF', [0, 200], "DEX $MM"), use_container_width=True)
    st.plotly_chart(create_ts_chart('CNV', '#FF00FF', [-200, 400], "CNV $MM"), use_container_width=True)

    st.markdown("### Greek Sensitivity Matrix")
    col_v, col_c = st.columns(2)
    with col_v:
        fig_v = px.bar(plot_df, x='strike', y='vega', title="VEGA", color_discrete_sequence=['#00FFFF'])
        st.plotly_chart(fig_v.update_layout(template="plotly_dark", height=300), use_container_width=True)
    with col_c:
        fig_c = px.line(plot_df, x='strike', y='charm', title="CHARM", color_discrete_sequence=['#FF00FF'])
        st.plotly_chart(fig_c.update_layout(template="plotly_dark", height=300), use_container_width=True)
