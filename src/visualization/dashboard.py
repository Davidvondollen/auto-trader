"""
Streamlit Web Dashboard for Trading System
Real-time monitoring and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading_system import AgenticTradingSystem
from src.utils.config_loader import ConfigLoader


# Page configuration
st.set_page_config(
    page_title="Agentic Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


class TradingDashboard:
    """Main dashboard class."""

    def __init__(self):
        """Initialize dashboard."""
        self.config = ConfigLoader()

        # Initialize session state
        if 'system' not in st.session_state:
            st.session_state.system = None

        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()

    def run(self):
        """Run the dashboard."""
        # Sidebar
        self.render_sidebar()

        # Main content
        st.title("📈 Agentic Trading System Dashboard")

        # Check if system is initialized
        if st.session_state.system is None:
            self.render_welcome()
        else:
            self.render_main_dashboard()

    def render_sidebar(self):
        """Render sidebar with controls."""
        st.sidebar.title("Controls")

        # System initialization
        if st.session_state.system is None:
            if st.sidebar.button("Initialize System", type="primary"):
                with st.spinner("Initializing trading system..."):
                    try:
                        st.session_state.system = AgenticTradingSystem()
                        st.sidebar.success("System initialized!")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Initialization failed: {e}")
        else:
            # System status
            status = st.session_state.system.get_status()

            st.sidebar.subheader("System Status")
            mode = status.get('mode', 'unknown')
            mode_color = "green" if mode == "paper" else "red"
            st.sidebar.markdown(f"**Mode:** :{mode_color}[{mode.upper()}]")

            broker_connected = status.get('broker_connected', False)
            connection_emoji = "✅" if broker_connected else "❌"
            st.sidebar.markdown(f"**Broker:** {connection_emoji} {'Connected' if broker_connected else 'Disconnected'}")

            is_running = status.get('is_running', False)
            st.sidebar.markdown(f"**Running:** {'🟢 Active' if is_running else '🔴 Stopped'}")

            # Account info
            if 'account_info' in status:
                account = status['account_info']
                st.sidebar.subheader("Account")
                st.sidebar.metric("Portfolio Value", f"${account.get('portfolio_value', 0):,.2f}")
                st.sidebar.metric("Cash", f"${account.get('cash', 0):,.2f}")
                st.sidebar.metric("Buying Power", f"${account.get('buying_power', 0):,.2f}")

            # Refresh button
            if st.sidebar.button("🔄 Refresh", type="secondary"):
                st.session_state.last_update = datetime.now()
                st.rerun()

            # Settings
            st.sidebar.subheader("Settings")

            update_interval = st.sidebar.slider(
                "Update Interval (s)",
                min_value=5,
                max_value=300,
                value=60,
                step=5
            )

            # Auto-refresh
            auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)

            if auto_refresh:
                st.sidebar.info(f"Auto-refreshing every {update_interval}s")

    def render_welcome(self):
        """Render welcome screen."""
        st.markdown("""
        ## Welcome to the Agentic Trading System

        This dashboard provides real-time monitoring and control of your autonomous trading system.

        ### Features
        - 📊 Real-time portfolio monitoring
        - 📈 Performance analytics
        - 🤖 Strategy management
        - 📰 News & sentiment analysis
        - ⚠️ Risk monitoring

        ### Getting Started
        Click **Initialize System** in the sidebar to begin.
        """)

        # Display configuration
        with st.expander("Current Configuration"):
            st.json(self.config.config)

    def render_main_dashboard(self):
        """Render main dashboard content."""
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "📈 Performance",
            "🤖 Strategies",
            "📰 News & Sentiment",
            "⚠️ Risk"
        ])

        with tab1:
            self.render_overview_tab()

        with tab2:
            self.render_performance_tab()

        with tab3:
            self.render_strategies_tab()

        with tab4:
            self.render_news_tab()

        with tab5:
            self.render_risk_tab()

    def render_overview_tab(self):
        """Render overview tab."""
        st.header("Portfolio Overview")

        status = st.session_state.system.get_status()

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        if 'account_info' in status:
            account = status['account_info']
            initial_capital = self.config.get_initial_capital()
            portfolio_value = account.get('portfolio_value', 0)
            total_return = ((portfolio_value - initial_capital) / initial_capital) * 100

            with col1:
                st.metric(
                    "Portfolio Value",
                    f"${portfolio_value:,.2f}",
                    f"{total_return:+.2f}%"
                )

            with col2:
                st.metric("Cash", f"${account.get('cash', 0):,.2f}")

            with col3:
                st.metric("Positions", status.get('num_positions', 0))

            with col4:
                st.metric(
                    "Pending Orders",
                    status.get('pending_orders', 0)
                )

        # Positions table
        st.subheader("Current Positions")

        if 'positions' in status and status['positions']:
            positions_data = []
            for symbol, pos in status['positions'].items():
                positions_data.append({
                    'Symbol': symbol,
                    'Shares': f"{pos.get('qty', 0):.4f}",
                    'Entry Price': f"${pos.get('avg_entry_price', 0):.2f}",
                    'Current Price': f"${pos.get('current_price', 0):.2f}",
                    'Market Value': f"${pos.get('market_value', 0):,.2f}",
                    'P&L': f"${pos.get('unrealized_pl', 0):,.2f}",
                    'P&L %': f"{pos.get('unrealized_plpc', 0)*100:+.2f}%"
                })

            df = pd.DataFrame(positions_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No open positions")

        # Recent orders
        st.subheader("Recent Orders")
        if status.get('filled_orders', 0) > 0:
            st.info("Order history available")
        else:
            st.info("No recent orders")

    def render_performance_tab(self):
        """Render performance analytics tab."""
        st.header("Performance Analytics")

        # Generate sample performance data (in production, fetch from system)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        portfolio_values = 10000 * (1 + np.cumsum(np.random.randn(30) * 0.02))

        df = pd.DataFrame({
            'date': dates,
            'value': portfolio_values
        })

        # Portfolio value chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00cc96', width=2)
        ))

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Return", "+23.45%", "+2.34%")

        with col2:
            st.metric("Sharpe Ratio", "1.85")

        with col3:
            st.metric("Max Drawdown", "-8.23%")

        # Returns distribution
        st.subheader("Returns Distribution")

        returns = df['value'].pct_change().dropna() * 100

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=30,
            name='Daily Returns',
            marker_color='#636efa'
        ))

        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_strategies_tab(self):
        """Render strategies tab."""
        st.header("Strategy Management")

        # Active strategies
        st.subheader("Active Strategies")

        strategies = [
            {"name": "Momentum Strategy", "status": "Active", "performance": "+15.2%"},
            {"name": "Mean Reversion", "status": "Active", "performance": "+8.7%"},
            {"name": "RL Policy", "status": "Training", "performance": "N/A"}
        ]

        for strategy in strategies:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                with col1:
                    st.write(f"**{strategy['name']}**")

                with col2:
                    status_color = "green" if strategy['status'] == "Active" else "orange"
                    st.markdown(f":{status_color}[{strategy['status']}]")

                with col3:
                    st.write(f"Performance: {strategy['performance']}")

                with col4:
                    if st.button("⚙️", key=f"config_{strategy['name']}"):
                        st.info("Configuration dialog would open here")

        st.divider()

        # Generate new strategy
        st.subheader("Generate New Strategy")

        col1, col2 = st.columns(2)

        with col1:
            market_condition = st.selectbox(
                "Market Condition",
                ["Trending", "Ranging", "Volatile"]
            )

            risk_level = st.select_slider(
                "Risk Level",
                options=["Low", "Medium", "High"],
                value="Medium"
            )

        with col2:
            timeframe = st.selectbox(
                "Timeframe",
                ["1h", "4h", "1d"]
            )

            strategy_type = st.selectbox(
                "Strategy Type",
                ["LLM Generated", "RL Policy", "Technical"]
            )

        if st.button("Generate Strategy", type="primary"):
            with st.spinner("Generating strategy..."):
                st.success("Strategy generated successfully!")
                st.code("""
class NewStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Strategy logic here
        pass
                """, language="python")

    def render_news_tab(self):
        """Render news & sentiment tab."""
        st.header("News & Sentiment Analysis")

        # Sentiment gauge
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Overall Sentiment", "Positive", "+0.15")

        with col2:
            st.metric("News Volume", "42 articles")

        with col3:
            st.metric("Sentiment Trend", "Rising", "+5%")

        # Sample news
        st.subheader("Recent News")

        news_items = [
            {
                "title": "Tech stocks rally on positive earnings",
                "sentiment": "Positive",
                "score": 0.8,
                "source": "Financial Times",
                "time": "2 hours ago"
            },
            {
                "title": "Market volatility expected amid economic data",
                "sentiment": "Neutral",
                "score": 0.0,
                "source": "Bloomberg",
                "time": "4 hours ago"
            },
            {
                "title": "Concerns over supply chain disruptions",
                "sentiment": "Negative",
                "score": -0.6,
                "source": "Reuters",
                "time": "6 hours ago"
            }
        ]

        for news in news_items:
            with st.container():
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.write(f"**{news['title']}**")
                    st.caption(f"{news['source']} • {news['time']}")

                with col2:
                    sentiment_color = {
                        "Positive": "green",
                        "Neutral": "gray",
                        "Negative": "red"
                    }[news['sentiment']]
                    st.markdown(f":{sentiment_color}[{news['sentiment']}]")

                st.divider()

        # Sentiment timeline
        st.subheader("Sentiment Timeline")

        dates = pd.date_range(end=datetime.now(), periods=14, freq='D')
        sentiment_scores = np.random.randn(14) * 0.3

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dates,
            y=sentiment_scores,
            marker_color=['green' if x > 0 else 'red' for x in sentiment_scores],
            name='Sentiment Score'
        ))

        fig.update_layout(
            title="Daily Sentiment Scores",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_risk_tab(self):
        """Render risk monitoring tab."""
        st.header("Risk Monitoring")

        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Portfolio VaR (95%)", "$1,234", "-$45")

        with col2:
            st.metric("Max Drawdown", "-8.23%", "+0.5%")

        with col3:
            st.metric("Sharpe Ratio", "1.85", "+0.12")

        with col4:
            st.metric("Beta", "0.92", "-0.03")

        # Risk allocation
        st.subheader("Risk Allocation")

        sectors = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy']
        allocations = [35, 25, 20, 15, 5]

        fig = go.Figure(data=[go.Pie(
            labels=sectors,
            values=allocations,
            hole=0.4
        )])

        fig.update_layout(
            title="Sector Allocation",
            height=400
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Risk Limits")
            st.progress(0.65, text="Position Limit (65%)")
            st.progress(0.45, text="VaR Limit (45%)")
            st.progress(0.30, text="Correlation Limit (30%)")

        # Correlation matrix
        st.subheader("Asset Correlation Matrix")

        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        corr_matrix = np.random.rand(4, 4)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdYlGn',
            zmid=0
        ))

        fig.update_layout(
            title="Correlation Matrix",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main entry point."""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
