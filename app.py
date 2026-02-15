import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 1. SYSTEM CONFIG & BRANDING
st.set_page_config(page_title="GEXRADAR // QUANT TERMINAL", layout="wide")

# Handle Page State
if 'current_page' not in st.session_state: 
    st.session_state.current_page = "DASHBOARD"

# URL Parameter Handling
params = st.query_params
if "page" in params:
    st.session_state.current_page = params["page"]

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;700&display=swap');
        .block-container { padding-top: 1.5rem !important; padding-bottom: 0rem !important; }
        header {visibility: hidden;} 
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0E1117; color: #E0E0E0; }
        
        .header-container {
            display: flex;
            align-items: baseline; gap: 40px;
            padding-bottom: 15px; border-bottom: 1px solid #30363D; margin-bottom: 25px;
        }

        .logo-text { 
            font-family: 'JetBrains Mono', monospace; 
            font-size: 26px; font-weight: 700; color: #00C805 !important; 
            letter-spacing: -1px; text-decoration: none !important;
            text-shadow: 0 0 10px rgba(0, 200, 5, 0.4);
        }

        .nav-link {
            font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 400;
            color: #8B949E !important; text-decoration: none !important;
            text-transform: uppercase; transition: all 0.3s ease; cursor: pointer;
        }

        .nav-link:hover { color: #00C805 !important; text-shadow: 0 0 15px #00C805 !important; }

        div.stButton > button {
            background-color: #161B22 !important; border: 1px solid #30363D !important;
            color: #FFFFFF !important; font-family: 'Inter', sans-serif !important;
            border-radius: 4px !important; height: 38px !important;
            font-size: 11px !important; font-weight: 700 !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important; width: 100% !important;
        }
        div.stButton > button:hover {
            border: 1px solid #00C805 !important; color: #00C805 !important;
            box-shadow: 0 0 20px rgba(0, 200, 5, 0.4) !important; transform: scale(1.02) !important;
        }

        .sidebar-label { font-size: 10px; color: #666; letter-spacing: 1.5px; font-weight: 800; text-transform: uppercase; margin-top: 20px; border-bottom: 1px solid #333; padding-bottom: 3px; }
        .data-block { background: rgba(255, 255, 255, 0.03); border: 1px solid #30363D; padding: 12px; border-radius: 4px; margin-bottom: 8px; }
        .data-label { font-size: 10px; color: #8B949E; margin-bottom: 2px; }
        .data-value { font-family: 'JetBrains Mono'; font-size: 16px; font-weight: 700; }
        .regime-container { padding: 20px; border-radius: 4px; border: 1px solid #30363D; background: #161B22; margin-bottom: 25px; }
        
        .ops-card { background: #161B22; border: 1px solid #30363D; padding: 25px; border-radius: 4px; height: 100%; border-left: 3px solid #00C805; }
        .ops-title { font-family: 'JetBrains Mono'; font-size: 18px; font-weight: 700; color: #FFFFFF; margin-bottom: 5px; }
        .ops-tag { font-family: 'JetBrains Mono'; color: #00C805; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 15px; opacity: 0.8; }
        .ops-label { font-size: 11px; color: #555; text-transform: uppercase; font-weight: 800; margin-top: 20px; letter-spacing: 1px; }
        .ops-content { font-size: 13px; color: #B0B0B0; line-height: 1.6; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 2. GLOBAL HEADER
st.markdown(f"""
    <div class="header-container">
        <a href="/?page=DASHBOARD" target="_self" class="logo-text">GEXRADAR<span style="color:white">_</span></a>
        <a href="/?page=DASHBOARD" target="_self" class="nav-link">Dashboard</a>
        <a href="/?page=OPERATIONS" target="_self" class="nav-link">Operations</a>
        <a href="/?page=ABOUT" target="_self" class="nav-link">About</a>
    </div>
""", unsafe_allow_html=True)

# 3. DATA ENGINE
st.sidebar.markdown("<p class='sidebar-label'>Terminal Configuration</p>", unsafe_allow_html=True)
asset_toggle = st.sidebar.radio("Select Asset", ["SPY", "QQQ"], horizontal=True)

if asset_toggle == "SPY":
    strikes = np.arange(580, 605, 0.5)
    spot_price = 592.40
    basis = 22.45
    gamma_flip_level = 591.0
    momentum_wall = 595.0
    max_pain = 590.0
    vol_trigger = 588.5
    equiv_label = "ES Equiv"
    equiv_mult = 10
    vvix = 84.20
    iv_rv_spread = 2.45
    flow_ratio = 0.62 # Simulated SPY Flow
else:
    strikes = np.arange(495, 520, 0.5)
    spot_price = 508.40
    basis = 145.00 
    gamma_flip_level = 506.5
    momentum_wall = 512.0
    max_pain = 505.0
    vol_trigger = 502.0
    equiv_label = "NQ Equiv"
    equiv_mult = 47.5
    vvix = 92.15
    iv_rv_spread = -1.12
    flow_ratio = 0.44 # Simulated QQQ Flow

# Time-series generation
times = pd.date_range(start='9:30', periods=30, freq='10min')
price_walk = spot_price + np.cumsum(np.random.normal(0, 0.4, 30))
t_series = pd.DataFrame({'Time': times, 'Price': price_walk, 'GEX': np.random.uniform(-400, 900, 30), 'DEX': np.random.uniform(20, 150, 30), 'CNV': np.random.uniform(-150, 350, 30)})

data = []
for s in strikes:
    is_major = 3.5 if s % 5 == 0 else (1.8 if s % 2.5 == 0 else 0.6)
    dist = abs(s - spot_price)
    call_g = np.exp(-(abs(s - (spot_price + 3))**2)/12) * is_major
    put_g = np.exp(-(abs(s - (spot_price - 4))**2)/12) * is_major
    es_val = (s * equiv_mult) + basis
    
    # Second-order Greeks (Simulated)
    vomma = (call_g + put_g) * (dist/spot_price) * 10
    zomma = (call_g - put_g) * (1 / (dist + 0.1))
    velocity = np.random.normal(0, 5) # Net change in GEX per tick
    
    data.append({
        'strike': s, 'es_strike': es_val,
        'call_gex': call_g * 6000, 'put_gex': -put_g * 6000, 
        'vanna': (call_g + put_g) * 0.4, 'oi': int((call_g + put_g) * 60000), 
        'vol_call': np.random.randint(1500, 9000) * call_g, 
        'vol_put': np.random.randint(1500, 9000) * put_g, 
        'vega': (call_g + put_g) * 0.25, 'charm': (call_g - put_g) * 0.12, 
        'iv': 0.15 + (dist**2 * 0.0008),
        'vomma': vomma, 'zomma': zomma, 'velocity': velocity
    })
df = pd.DataFrame(data)

# Regime Logic Calculation
is_long_gamma = spot_price > gamma_flip_level
regime_color = "#00C805" if is_long_gamma else "#FF3B3B"
regime_label = "STABLE / LONG GAMMA" if is_long_gamma else "VOLATILE / SHORT GAMMA"

# Flowing Structural Bias Logic
total_vomma = df["vomma"].sum()
if is_long_gamma:
    bias_note = "ACCUMULATE / BUY DIPS"
    bias_color = "#00FF00"
elif not is_long_gamma and total_vomma > 5:
    bias_note = "PROTECTIVE / LONG VOL"
    bias_color = "#FF3B3B"
else:
    bias_note = "NEUTRAL / SCALP ONLY"
    bias_color = "#FFD700"

# 4. DASHBOARD PAGE
if st.session_state.current_page == "DASHBOARD":
    st.sidebar.markdown("<p class='sidebar-label'>Structural Context</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="data-block"><div class="data-label">Vomma (Vol Sens)</div><div class="data-value">{df["vomma"].sum():.2f}</div></div><div class="data-block"><div class="data-label">Zomma (Gamma Accel)</div><div class="data-value">{df["zomma"].sum():.2f}</div></div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("<p class='sidebar-label'>Regime Indicators</p>", unsafe_allow_html=True)
    # Flow Ratio Added Back
    fr_color = "#00C805" if flow_ratio > 0.50 else "#FF3B3B"
    st.sidebar.markdown(f'<div class="data-block"><div class="data-label">Flow Ratio</div><div class="data-value" style="color:{fr_color}">{flow_ratio:.2f}</div></div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="data-block"><div class="data-label">VVIX (Vol of Vol)</div><div class="data-value" style="color:#00FFFF">{vvix}</div></div><div class="data-block"><div class="data-label">IV-RV Spread</div><div class="data-value">{iv_rv_spread}%</div></div>', unsafe_allow_html=True)

    st.sidebar.markdown("<p class='sidebar-label'>Topography Scan</p>", unsafe_allow_html=True)
    topo = df[(df['strike'] > spot_price - 8) & (df['strike'] < spot_price + 8)].sort_values('strike')
    z_topo = np.outer(np.linspace(1, 0.1, 10), (np.abs(topo['call_gex']) + np.abs(topo['put_gex'])).values)
    st.sidebar.plotly_chart(go.Figure(data=[go.Surface(z=z_topo, x=topo['strike'].values, colorscale='Viridis', showscale=False)]).update_layout(height=180, margin=dict(l=0,r=0,b=0,t=0), template="plotly_dark"), use_container_width=True)

    # UPDATED: Flowing Regime Header
    st.markdown(f"""
        <div class="regime-container" style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 11px; color: #808495; text-transform: uppercase;">Market Regime: {asset_toggle}</div>
                <div style="font-family: 'JetBrains Mono'; font-size: 24px; font-weight: 700; color: {regime_color};">{regime_label}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 10px; color: #808495; letter-spacing: 1px;">STRUCTURAL BIAS</div>
                <div style="font-family: 'JetBrains Mono'; font-size: 18px; font-weight: 700; color: {bias_color};">{bias_note}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if 'radar_mode' not in st.session_state: st.session_state.radar_mode = "GEX"
    m_cols = st.columns(7) 
    modes = ["GEX", "VOL", "HEAT", "VELOCITY", "SURF", "SMILE", "DELTA"]
    labels = ["OI / GEX", "VOLUME", "HEATMAP", "VELOCITY", "SURFACE", "SMILE", "NET DELTA"]
    
    for col, mode, label in zip(m_cols, modes, labels):
        with col:
            btn_text = f"● {label}" if st.session_state.radar_mode == mode else label
            if st.button(btn_text, key=f"mode_{mode}"):
                st.session_state.radar_mode = mode
                st.rerun()

    plot_df = df[(df['strike'] >= min(strikes)) & (df['strike'] <= max(strikes))]
    
    if st.session_state.radar_mode == "VELOCITY":
        dual_vel = st.toggle("Dual View (SPY + QQQ Velocity)", value=False)
        if dual_vel:
            fig_main = make_subplots(rows=1, cols=2, subplot_titles=("SPY VELOCITY", "QQQ VELOCITY"), horizontal_spacing=0.1)
            for i, (asset, range_s) in enumerate([("SPY", np.arange(580, 605, 0.5)), ("QQQ", np.arange(495, 520, 0.5))], 1):
                v_data = np.random.uniform(-10, 10, (len(range_s), 10))
                v_text = [[f"{v:+.1f}" if abs(v) > 7.5 else "" for v in r] for r in v_data]
                fig_main.add_trace(go.Heatmap(z=v_data, y=range_s, colorscale='RdYlGn', text=v_text, texttemplate="%{text}", showscale=(i==2)), row=1, col=i)
            fig_main.update_layout(xaxis_title="LOOKBACK", yaxis_title="STRIKE", xaxis2_title="LOOKBACK")
        else:
            vel_data = np.random.uniform(-10, 10, (len(plot_df), 10))
            text_vals = [[f"{val:+.1f}" if abs(val) > 7.5 else "" for val in row] for row in vel_data]
            fig_main = go.Figure(data=go.Heatmap(z=vel_data, y=plot_df['strike'], colorscale='RdYlGn', text=text_vals, texttemplate="%{text}", hovertemplate="Strike: %{y}<br>Velocity: %{z:.2f}<extra></extra>"))
            fig_main.update_layout(xaxis_title="LOOKBACK PERIODS", yaxis_title="STRIKE PRICE", yaxis=dict(dtick=1))

    elif st.session_state.radar_mode == "SURF":
        days = np.array([1, 7, 30, 60, 90])
        z_vol = np.array([plot_df['iv'].values * (1 + 0.05 * np.log(d)) for d in days])
        fig_main = go.Figure(data=[go.Surface(z=z_vol, x=plot_df['strike'], y=days, colorscale='Thermal', customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{x}}<br>{equiv_label}: %{{customdata:.2f}}<br>DTE: %{{y}}<br>IV: %{{z:.2f}}<extra></extra>")])
        fig_main.update_layout(scene=dict(xaxis_title="STRIKE", yaxis_title="DTE", zaxis_title="IV"))
    elif st.session_state.radar_mode == "SMILE":
        days = np.array([1, 5, 10, 20, 30])
        z_smile = np.array([0.12 + (abs(plot_df['strike'] - spot_price)**1.5 * 0.001) / np.sqrt(d) for d in days])
        fig_main = go.Figure(data=[go.Surface(z=z_smile, x=plot_df['strike'], y=days, colorscale='IceFire', customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{x}}<br>{equiv_label}: %{{customdata:.2f}}<br>DTE: %{{y}}<br>Vol: %{{z:.4f}}<extra></extra>")])
        fig_main.update_layout(scene=dict(xaxis_title="STRIKE", yaxis_title="DTE", zaxis_title="SMILE"))
    elif st.session_state.radar_mode == "DELTA":
        time_steps = np.arange(10)
        z_delta = np.outer(np.sin(time_steps/2), plot_df['call_gex'].values * 0.1)
        fig_main = go.Figure(data=[go.Surface(z=z_delta, x=plot_df['strike'], y=time_steps, colorscale='Portland', customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{x}}<br>{equiv_label}: %{{customdata:.2f}}<br>Time: %{{y}}<br>Delta: %{{z:.2f}}<extra></extra>")])
        fig_main.update_layout(scene=dict(xaxis_title="STRIKE", yaxis_title="TIME", zaxis_title="NET DELTA"))
    elif st.session_state.radar_mode == "HEAT":
        dual_heat = st.toggle("Dual View (SPY + QQQ Heatmap)", value=False)
        dtes = ["0DTE", "1DTE", "7DTE", "14DTE", "30DTE", "60DTE", "90DTE"]
        if dual_heat:
            fig_main = make_subplots(rows=1, cols=2, subplot_titles=("SPY GEX TERM", "QQQ GEX TERM"), horizontal_spacing=0.1)
            for i, (asset, range_s) in enumerate([("SPY", np.arange(580, 605, 0.5)), ("QQQ", np.arange(495, 520, 0.5))], 1):
                h_data = np.random.uniform(-500, 1500, (len(range_s), len(dtes)))
                h_text = [[f"{v:,.0f}" for v in r] for r in h_data]
                fig_main.add_trace(go.Heatmap(z=h_data, x=dtes, y=range_s, colorscale='Magma', text=h_text, texttemplate="%{text}", showscale=(i==2)), row=1, col=i)
            fig_main.update_layout(xaxis_title="DTE", yaxis_title="STRIKE", xaxis2_title="DTE")
        else:
            heat_data = np.random.uniform(-500, 1500, (len(plot_df), len(dtes)))
            text_vals = [[f"{val:,.0f}" for val in row] for row in heat_data]
            fig_main = go.Figure(data=go.Heatmap(z=heat_data, x=dtes, y=plot_df['strike'], colorscale='Magma', text=text_vals, texttemplate="%{text}", customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{y}}<br>{equiv_label}: %{{customdata:.2f}}<br>DTE: %{{x}}<br>GEX: %{{z:,.0f}}<extra></extra>"))
            fig_main.update_layout(xaxis_title="EXPIRATION (DTE)", yaxis_title="STRIKE PRICE", yaxis=dict(dtick=1))
    else:
        fig_main = go.Figure()
        weight_mode = st.toggle("Delta-Weighted GEX", value=False)
        w_mult = 1.0
        
        if st.session_state.radar_mode == "GEX":
            if weight_mode: w_mult = np.clip(1 - (abs(plot_df['strike'] - spot_price) / 10), 0.1, 1)
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['call_gex']*w_mult, orientation='h', marker=dict(color='#00C805', line=dict(color='#00FF00', width=1)), name="CALL GEX", customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{y}}<br>{equiv_label}: %{{customdata:.2f}}<br>Call GEX: %{{x:,.0f}}<extra></extra>"))
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['put_gex']*w_mult, orientation='h', marker=dict(color='#FF3B3B', line=dict(color='#FF5555', width=1)), name="PUT GEX", customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{y}}<br>{equiv_label}: %{{customdata:.2f}}<br>Put GEX: %{{x:,.0f}}<extra></extra>"))
            fig_main.update_layout(xaxis_title="NET GEX ($MM)", yaxis_title="STRIKE PRICE", bargap=0.1)
        else:
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['vol_call'], orientation='h', marker=dict(color='#00C805', line=dict(color='#00FF00', width=1)), name="CALL VOL", customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{y}}<br>{equiv_label}: %{{customdata:.2f}}<br>Call Vol: %{{x:,.0f}}<extra></extra>"))
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=-plot_df['vol_put'], orientation='h', marker=dict(color='#FF3B3B', line=dict(color='#FF5555', width=1)), name="PUT VOL", customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{y}}<br>{equiv_label}: %{{customdata:.2f}}<br>Put Vol: %{{x:,.0f}}<extra></extra>"))
            fig_main.update_layout(xaxis_title="VOLUME (CONTRACTS)", yaxis_title="STRIKE PRICE", bargap=0.1)
        
        levels = [(gamma_flip_level, "#808495", "FLIP", "dot"), (spot_price, "#FFF", "SPOT", "solid"), (momentum_wall, "#00FFFF", "MOM WALL", "dash"), (max_pain, "#FFD700", "MAX PAIN", "dot"), (vol_trigger, "#FF3B3B", "VOL TRIGGER", "dashdot")]
        for lvl, clr, txt, dash in levels:
            fig_main.add_hline(y=lvl, line_dash=dash, line_color=clr, annotation_text=f" {txt}", annotation_position="right")

    # Layout Adjustment for Title Visibility
    fig_main.update_layout(height=700, template="plotly_dark", showlegend=False, margin=dict(t=80, r=60, b=40, l=60), hovermode="closest")
    st.plotly_chart(fig_main, use_container_width=True)

    # --- KEY LEVELS TABLE ---
    st.markdown("### Key Levels Table")
    table_df = df.copy()
    table_df['Net GEX'] = table_df['call_gex'] + table_df['put_gex']
    table_df['Abs GEX'] = table_df['call_gex'].abs() + table_df['put_gex'].abs()
    table_df['Dist (%)'] = ((table_df['strike'] - spot_price) / spot_price) * 100
    
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        filter_type = st.selectbox("Rank By", ["Top Abs Exposure", "Top Call Walls", "Top Put Walls", "Closest to Price"])
    
    if filter_type == "Top Abs Exposure":
        display_df = table_df.sort_values('Abs GEX', ascending=False).head(10)
    elif filter_type == "Top Call Walls":
        display_df = table_df.sort_values('call_gex', ascending=False).head(10)
    elif filter_type == "Top Put Walls":
        display_df = table_df.sort_values('put_gex', ascending=True).head(10)
    else:
        table_df['dist_abs'] = table_df['strike'].sub(spot_price).abs()
        display_df = table_df.sort_values('dist_abs').head(10).drop(columns=['dist_abs'])

    display_df = display_df[['strike', 'es_strike', 'call_gex', 'put_gex', 'Net GEX', 'Abs GEX', 'Dist (%)', 'velocity']]
    display_df.columns = ['Strike', equiv_label, 'Call GEX', 'Put GEX', 'Net GEX', 'Abs GEX', 'Dist %', 'Velocity']
    
    def style_gex_rows(val):
        color = '#00C805' if val > 0 else '#FF3B3B' if val < 0 else '#8B949E'
        return f'color: {color}; font-weight: bold;'

    st.dataframe(
        display_df.style.format({
            'Strike': '{:.1f}', equiv_label: '{:.2f}', 'Call GEX': '{:,.0f}', 
            'Put GEX': '{:,.0f}', 'Net GEX': '{:,.0f}', 'Abs GEX': '{:,.0f}', 'Dist %': '{:+.2f}%', 'Velocity': '{:+.2f}'
        }).map(style_gex_rows, subset=['Net GEX']),
        use_container_width=True, hide_index=True
    )
    
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Export Levels to CSV", data=csv, file_name=f"{asset_toggle}_key_levels.csv", mime='text/csv')
    st.markdown("---")

    # --- INSTITUTIONAL TIME-SERIES ($MM) ---
    st.markdown("### Institutional Time-Series ($MM)")
    for col, color, title in [('GEX', '#00C805', 'GEX $MM'), ('DEX', '#00FFFF', 'DEX $MM'), ('CNV', '#FF00FF', 'CNV $MM')]:
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ts.add_trace(go.Scatter(x=t_series['Time'], y=t_series[col], line=dict(color=color, width=3)), secondary_y=False)
        fig_ts.add_trace(go.Scatter(x=t_series['Time'], y=t_series['Price'], line=dict(color='white', width=1, dash='dot'), opacity=0.4), secondary_y=True)
        st.plotly_chart(fig_ts.update_layout(height=280, template="plotly_dark", showlegend=False, title=title, margin=dict(t=30, b=20)), use_container_width=True)

    # --- GREEK SENSITIVITY MATRIX ---
    st.markdown("### Greek Sensitivity Matrix")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.bar(plot_df, x='strike', y='vomma', title="VOMMA (Vol Sens)", color_discrete_sequence=['#00FFFF']).update_layout(template="plotly_dark", height=300), use_container_width=True)
    with c2: st.plotly_chart(px.line(plot_df, x='strike', y='zomma', title="ZOMMA (Gamma Accel)", color_discrete_sequence=['#FF00FF']).update_layout(template="plotly_dark", height=300), use_container_width=True)

elif st.session_state.current_page == "OPERATIONS":
    st.markdown("<h2 style='font-family: JetBrains Mono; color:#00C805;'>QUANTITATIVE SPECIFICATIONS</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="ops-card"><div class="ops-tag">Instrument_01</div><div class="ops-title">VOMMA</div>', unsafe_allow_html=True)
        st.latex(r"Vomma = \frac{\partial \text{Vega}}{\partial \sigma}")
        st.markdown('<div class="ops-content">Measures convexity of Vega. Critical for monitoring "Vol of Vol" spikes.</div></div><br>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="ops-card"><div class="ops-tag">Instrument_02</div><div class="ops-title">ZOMMA</div>', unsafe_allow_html=True)
        st.latex(r"Zomma = \frac{\partial \Gamma}{\partial \sigma}")
        st.markdown('<div class="ops-content">Measures sensitivity of Gamma to volatility changes. Essential for timing regime shifts.</div></div><br>', unsafe_allow_html=True)

elif st.session_state.current_page == "ABOUT":
    st.title("GEXRADAR QUANT TERMINAL v4.5")
    st.write("Proprietary Liquidity & Reflexivity Analysis Suite.")
