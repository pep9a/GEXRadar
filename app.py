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
        
        /* ENHANCED OPS GRID */
        .ops-card { background: #161B22; border: 1px solid #30363D; padding: 25px; border-radius: 4px; height: 100%; border-left: 3px solid #00C805; }
        .ops-title { font-family: 'JetBrains Mono'; font-size: 18px; font-weight: 700; color: #FFFFFF; margin-bottom: 5px; }
        .ops-tag { font-family: 'JetBrains Mono'; color: #00C805; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 15px; opacity: 0.8; }
        .ops-label { font-size: 11px; color: #555; text-transform: uppercase; font-weight: 800; margin-top: 20px; letter-spacing: 1px; }
        .ops-content { font-size: 13px; color: #B0B0B0; line-height: 1.6; margin-top: 5px; }
        .math-box { background: #0E1117; padding: 15px; border-radius: 4px; border: 1px solid #222; margin: 10px 0; }
        .disclaimer-box { font-size: 11px; color: #444; margin-top: 50px; border-top: 1px solid #222; padding-top: 20px; line-height: 1.5; }
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

# 3. DATA ENGINE (MODIFIED FOR NQ TOGGLE)
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
else:
    # QQQ logic with NQ correct pricing (approx 24k)
    strikes = np.arange(495, 520, 0.5)
    spot_price = 508.40
    basis = 145.00 
    gamma_flip_level = 506.5
    momentum_wall = 512.0
    max_pain = 505.0
    vol_trigger = 502.0
    equiv_label = "NQ Equiv"
    equiv_mult = 47.5 # NQ is roughly 47-48x QQQ price

times = pd.date_range(start='9:30', periods=30, freq='10min')
price_walk = spot_price + np.cumsum(np.random.normal(0, 0.4, 30))
t_series = pd.DataFrame({'Time': times, 'Price': price_walk, 'GEX': np.random.uniform(-400, 900, 30), 'DEX': np.random.uniform(20, 150, 30), 'CNV': np.random.uniform(-150, 350, 30)})

data = []
for s in strikes:
    is_major = 3.5 if s % 5 == 0 else (1.8 if s % 2.5 == 0 else 0.6)
    call_g = np.exp(-(abs(s - (spot_price + 3))**2)/12) * is_major
    put_g = np.exp(-(abs(s - (spot_price - 4))**2)/12) * is_major
    es_val = (s * equiv_mult) + basis
    data.append({
        'strike': s, 
        'es_strike': es_val,
        'call_gex': call_g * 6000, 
        'put_gex': -put_g * 6000, 
        'vanna': (call_g + put_g) * 0.4, 
        'oi': int((call_g + put_g) * 60000), 
        'vol_call': np.random.randint(1500, 9000) * call_g, 
        'vol_put': np.random.randint(1500, 9000) * put_g, 
        'vega': (call_g + put_g) * 0.25, 
        'charm': (call_g - put_g) * 0.12, 
        'iv': 0.15 + (abs(s - spot_price)**2 * 0.0008)
    })
df = pd.DataFrame(data)

# FIX: Flow Ratio Color Logic
flow_ratio_val = df["vol_call"].sum()/(df["vol_call"].sum()+df["vol_put"].sum())
flow_color = "#00C805" if flow_ratio_val >= 0.50 else "#FF3B3B"

# REGIME ADJUSTMENT LOGIC
is_long_gamma = spot_price > gamma_flip_level
regime_color = "#00C805" if is_long_gamma else "#FF3B3B"
regime_label = "STABLE / LONG GAMMA" if is_long_gamma else "VOLATILE / SHORT GAMMA"

# 4. DASHBOARD PAGE
if st.session_state.current_page == "DASHBOARD":
    st.sidebar.markdown("<p class='sidebar-label'>Structural Context</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="data-block"><div class="data-label">Flow Ratio</div><div class="data-value" style="color:{flow_color}">{flow_ratio_val:.2f}</div></div><div class="data-block"><div class="data-label">Momentum Wall</div><div class="data-value">${momentum_wall}</div></div><div class="data-block"><div class="data-label">Max Pain</div><div class="data-value">${max_pain}</div></div><div class="data-block"><div class="data-label">Zero Gamma</div><div class="data-value">${gamma_flip_level}</div></div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("<p class='sidebar-label'>Risk & Volatility</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="data-block"><div class="data-label">Vol Trigger</div><div class="data-value" style="color:#FF3B3B">${vol_trigger}</div></div><div class="data-block"><div class="data-label">Vanna Exposure</div><div class="data-value">${df["vanna"].sum():.2f}M</div></div><div class="data-block"><div class="data-label">Total OI</div><div class="data-value">{df["oi"].sum():,}</div></div>', unsafe_allow_html=True)

    st.sidebar.markdown("<p class='sidebar-label'>Topography Scan</p>", unsafe_allow_html=True)
    topo = df[(df['strike'] > spot_price - 8) & (df['strike'] < spot_price + 8)].sort_values('strike')
    z_topo = np.outer(np.linspace(1, 0.1, 10), (np.abs(topo['call_gex']) + np.abs(topo['put_gex'])).values)
    st.sidebar.plotly_chart(go.Figure(data=[go.Surface(z=z_topo, x=topo['strike'].values, colorscale='Viridis', showscale=False)]).update_layout(height=180, margin=dict(l=0,r=0,b=0,t=0), template="plotly_dark"), use_container_width=True)

    # REGIME BANNER RENDERING
    st.markdown(f'<div class="regime-container"><div style="font-size: 11px; color: #808495; text-transform: uppercase;">Market Regime: {asset_toggle}</div><div style="font-family: \'JetBrains Mono\'; font-size: 24px; font-weight: 700; color: {regime_color};">{regime_label}</div></div>', unsafe_allow_html=True)

    if 'radar_mode' not in st.session_state: st.session_state.radar_mode = "GEX"
    m_cols = st.columns(6)
    modes = ["GEX", "VOL", "HEAT", "SURF", "SMILE", "DELTA"]
    labels = ["OI / GEX", "VOLUME", "HEATMAP", "SURFACE", f"{asset_toggle} SMILE", "NET DELTA"]
    
    for col, mode, label in zip(m_cols, modes, labels):
        with col:
            btn_text = f"● {label}" if st.session_state.radar_mode == mode else label
            if st.button(btn_text, key=f"mode_{mode}"):
                st.session_state.radar_mode = mode
                st.rerun()

    # Dynamic Frame Adjustment: Ensures lines are always visible on the chart
    chart_min = min(strikes)
    chart_max = max(strikes)
    plot_df = df[(df['strike'] >= chart_min) & (df['strike'] <= chart_max)]
    
    if st.session_state.radar_mode == "SURF":
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
        dtes = ["0DTE", "1DTE", "7DTE", "14DTE", "30DTE", "60DTE", "90DTE"]
        strikes_short = plot_df['strike'].values
        heat_data = np.random.uniform(-500, 1500, (len(strikes_short), len(dtes)))
        text_vals = [[f"{val:,.0f}" for val in row] for row in heat_data]
        fig_main = go.Figure(data=go.Heatmap(z=heat_data, x=dtes, y=strikes_short, colorscale='Magma', text=text_vals, texttemplate="%{text}", customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{y}}<br>{equiv_label}: %{{customdata:.2f}}<br>DTE: %{{x}}<br>GEX: %{{z:,.0f}}<extra></extra>"))
        fig_main.update_layout(xaxis_title="EXPIRATION (DTE)", yaxis_title="STRIKE PRICE", yaxis=dict(dtick=1))
    else:
        fig_main = go.Figure()
        if st.session_state.radar_mode == "GEX":
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['call_gex'], orientation='h', marker=dict(color='#00C805', line=dict(color='#00FF00', width=1)), name="CALL GEX", customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{y}}<br>{equiv_label}: %{{customdata:.2f}}<br>Call GEX: %{{x:,.0f}}<extra></extra>"))
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['put_gex'], orientation='h', marker=dict(color='#FF3B3B', line=dict(color='#FF5555', width=1)), name="PUT GEX", customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{y}}<br>{equiv_label}: %{{customdata:.2f}}<br>Put GEX: %{{x:,.0f}}<extra></extra>"))
            fig_main.update_layout(xaxis_title="NET GEX ($MM)", yaxis_title="STRIKE PRICE", bargap=0.1)
        else:
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=plot_df['vol_call'], orientation='h', marker=dict(color='#00C805', line=dict(color='#00FF00', width=1)), name="CALL VOL", customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{y}}<br>{equiv_label}: %{{customdata:.2f}}<br>Call Vol: %{{x:,.0f}}<extra></extra>"))
            fig_main.add_trace(go.Bar(y=plot_df['strike'], x=-plot_df['vol_put'], orientation='h', marker=dict(color='#FF3B3B', line=dict(color='#FF5555', width=1)), name="PUT VOL", customdata=plot_df['es_strike'], hovertemplate=f"Strike: %{{y}}<br>{equiv_label}: %{{customdata:.2f}}<br>Put Vol: %{{x:,.0f}}<extra></extra>"))
            fig_main.update_layout(xaxis_title="VOLUME (CONTRACTS)", yaxis_title="STRIKE PRICE", bargap=0.1)
        
        # HORIZONTAL LEVEL MARKERS - PINNED AND LABELED
        levels = [
            (gamma_flip_level, "#808495", "FLIP", "dot"), 
            (spot_price, "#FFF", "SPOT", "solid"), 
            (momentum_wall, "#00FFFF", "MOM WALL", "dash"), 
            (max_pain, "#FFD700", "MAX PAIN", "dot"), 
            (vol_trigger, "#FF3B3B", "VOL TRIGGER", "dashdot")
        ]
        for lvl, clr, txt, dash in levels:
            fig_main.add_hline(y=lvl, line_dash=dash, line_color=clr, annotation_text=f" {txt}", annotation_position="right")

    fig_main.update_layout(height=700, template="plotly_dark", showlegend=False, margin=dict(t=10, r=60), hovermode="closest")
    st.plotly_chart(fig_main, use_container_width=True)

# 5. OPERATIONS PAGE (UNCHANGED)
elif st.session_state.current_page == "OPERATIONS":
    st.markdown("<h2 style='font-family: JetBrains Mono; color:#00C805;'>QUANTITATIVE SPECIFICATIONS</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="ops-card"><div class="ops-tag">Instrument_01</div><div class="ops-title">NET GAMMA EXPOSURE (GEX)</div>', unsafe_allow_html=True)
        st.latex(r"GEX = \sum \left( OI \cdot \Gamma \cdot 100 \cdot S^2 \right)")
        st.markdown('<div class="ops-label">Operational Thresholds</div><div class="ops-content"><b>Positive GEX:</b> Structural stability. <br><b>Negative GEX:</b> Structural instability.</div></div><br>', unsafe_allow_html=True)
        st.markdown('<div class="ops-card"><div class="ops-tag">Instrument_03</div><div class="ops-title">GAMMA FLIP & VOL TRIGGER</div>', unsafe_allow_html=True)
        st.latex(r"\Phi_{0} \rightarrow \sum \Gamma_{C} - \sum \Gamma_{P} = 0")
        st.markdown('<div class="ops-label">Operational Thresholds</div><div class="ops-content"><b>Above Gamma Flip:</b> Long-bias. <br><b>Below Gamma Flip:</b> Short-bias.</div></div><br>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="ops-card"><div class="ops-tag">Instrument_02</div><div class="ops-title">FLOW RATIO (AGGRESSION)</div>', unsafe_allow_html=True)
        st.latex(r"Ratio = \frac{\text{Vol}_{\text{Ask}}}{\text{Vol}_{\text{Total}}}")
        st.markdown('<div class="ops-label">Operational Thresholds</div><div class="ops-content"><b>> 0.70:</b> Extreme Bullish. <br><b>< 0.30:</b> Extreme Bearish.</div></div><br>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer-box"><b>QUANTITATIVE RISK DISCLOSURE:</b> Theoretical Greek modeling assumes standard Black-Scholes-Merton hedging behaviors.</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "ABOUT":
    st.title("GEXRADAR QUANT TERMINAL v4.4")
    st.write("Proprietary Liquidity & Reflexivity Analysis Suite.")
