import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from supply_chain_simulator import SupplyChainSimulator
from ai_models import AIDecisionEngine
from fraud_detector import FraudDetector
from rules_engine import RulesEngine
from blockchain_simulator import BlockchainSimulator

# Configure page
st.set_page_config(
    page_title="AI-Driven Healthcare Supply Chain Simulation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = SupplyChainSimulator()
    st.session_state.ai_engine = AIDecisionEngine()
    st.session_state.fraud_detector = FraudDetector()
    st.session_state.rules_engine = RulesEngine()
    st.session_state.blockchain = BlockchainSimulator()
    st.session_state.simulation_running = False

def main():
    st.title("🏥 AI-Driven Healthcare Supply Chain Simulation")
    st.markdown("### DBA Research: Intelligent Contract Automation for Healthcare Supply Chains")
    
    # Sidebar controls
    st.sidebar.title("Simulation Controls")
    
    # Simulation parameters
    st.sidebar.subheader("Parameters")
    num_drugs = st.sidebar.slider("Number of Drugs", 10, 100, 50)
    num_suppliers = st.sidebar.slider("Number of Suppliers", 3, 20, 10)
    num_hospitals = st.sidebar.slider("Number of Hospitals", 5, 50, 25)
    simulation_days = st.sidebar.slider("Simulation Period (days)", 30, 365, 90)
    
    # AI Model Configuration
    st.sidebar.subheader("AI Configuration")
    fraud_sensitivity = st.sidebar.slider("Fraud Detection Sensitivity", 0.1, 1.0, 0.7)
    restock_threshold = st.sidebar.slider("Restock Threshold (%)", 10, 50, 20)
    expiry_warning_days = st.sidebar.slider("Expiry Warning (days)", 7, 90, 30)
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🚀 Start Simulation"):
            st.session_state.simulation_running = True
            with st.spinner("Initializing simulation..."):
                st.session_state.simulator.initialize(
                    num_drugs=num_drugs,
                    num_suppliers=num_suppliers,
                    num_hospitals=num_hospitals,
                    days=simulation_days
                )
                st.session_state.ai_engine.configure(
                    fraud_sensitivity=fraud_sensitivity,
                    restock_threshold=restock_threshold,
                    expiry_warning_days=expiry_warning_days
                )
    
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.simulation_running = False
            st.rerun()
    
    if st.session_state.simulation_running:
        display_dashboard()
    else:
        display_welcome()

def display_welcome():
    st.markdown("""
    ## Welcome to the Healthcare Supply Chain AI Simulation
    
    This simulation demonstrates how AI-enhanced smart contracts can revolutionize healthcare supply chain management through:
    
    ### 🎯 Key Features
    - **Drug Expiration Tracking**: Real-time monitoring of drug shelf life
    - **Fraud Detection**: AI-powered anomaly detection for suspicious activities
    - **Predictive Restocking**: Machine learning-based inventory optimization
    - **Adaptive Prioritization**: Dynamic shipment scheduling based on urgency
    - **Blockchain Integration**: Immutable record keeping for audit trails
    
    ### 🔬 Research Applications
    - Demonstrate AI decision-making in supply chains
    - Analyze efficiency improvements through automation
    - Validate fraud detection algorithms
    - Measure cost reduction and waste minimization
    
    **Configure your simulation parameters in the sidebar and click "Start Simulation" to begin.**
    """)

def display_dashboard():
    # Generate current simulation data
    current_data = st.session_state.simulator.get_current_state()
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Drugs Tracked",
            f"{current_data['total_drugs']:,}",
            delta=f"+{current_data['new_drugs_today']}"
        )
    
    with col2:
        st.metric(
            "Active Suppliers",
            current_data['active_suppliers'],
            delta=f"{current_data['supplier_performance']:.1%}"
        )
    
    with col3:
        st.metric(
            "Fraud Alerts",
            current_data['fraud_alerts'],
            delta=f"+{current_data['new_fraud_alerts']}"
        )
    
    with col4:
        st.metric(
            "Expiring Soon",
            current_data['expiring_drugs'],
            delta=f"-{current_data['expired_prevented']}"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Inventory Dashboard",
        "🔍 Fraud Detection",
        "🤖 AI Decisions",
        "📈 Analytics",
        "🔗 Blockchain Audit"
    ])
    
    with tab1:
        display_inventory_dashboard(current_data)
    
    with tab2:
        display_fraud_detection(current_data)
    
    with tab3:
        display_ai_decisions(current_data)
    
    with tab4:
        display_analytics(current_data)
    
    with tab5:
        display_blockchain_audit(current_data)

def display_inventory_dashboard(data):
    st.subheader("📦 Real-time Inventory Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Inventory levels chart
        inventory_df = data['inventory_levels']
        fig = px.bar(
            inventory_df,
            x='drug_name',
            y='current_stock',
            color='stock_status',
            title="Current Inventory Levels by Drug",
            color_discrete_map={
                'Critical': '#ff4444',
                'Low': '#ffaa00',
                'Normal': '#00aa00',
                'Overstocked': '#0088ff'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Expiration alerts
        st.markdown("#### ⚠️ Expiration Alerts")
        expiring_df = data['expiring_drugs_detail']
        for _, row in expiring_df.iterrows():
            days_to_expire = row['days_to_expire']
            if days_to_expire <= 7:
                st.error(f"**{row['drug_name']}** expires in {days_to_expire} days")
            elif days_to_expire <= 30:
                st.warning(f"**{row['drug_name']}** expires in {days_to_expire} days")
    
    # AI Recommendations
    st.markdown("#### 🤖 AI Restocking Recommendations")
    recommendations = st.session_state.ai_engine.get_restock_recommendations(data)
    
    for rec in recommendations:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{rec['drug_name']}**")
        with col2:
            st.write(f"Recommended: {rec['recommended_quantity']}")
        with col3:
            if rec['urgency'] == 'High':
                st.error("🔴 High Priority")
            elif rec['urgency'] == 'Medium':
                st.warning("🟡 Medium Priority")
            else:
                st.info("🟢 Low Priority")

def display_fraud_detection(data):
    st.subheader("🔍 AI-Powered Fraud Detection System")
    
    # Fraud detection results
    fraud_results = st.session_state.fraud_detector.analyze_transactions(data['transactions'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Fraud risk timeline
        fig = px.line(
            fraud_results['timeline'],
            x='timestamp',
            y='fraud_score',
            title="Fraud Risk Score Over Time",
            color_discrete_sequence=['red']
        )
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                     annotation_text="Alert Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🚨 Active Alerts")
        for alert in fraud_results['alerts']:
            st.error(f"**{alert['type']}**: {alert['description']}")
            st.write(f"Risk Score: {alert['risk_score']:.2f}")
            st.write(f"Detected: {alert['timestamp']}")
            st.write("---")
    
    # Anomaly patterns
    st.markdown("#### 📊 Anomaly Patterns")
    anomaly_df = fraud_results['anomalies']
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            anomaly_df,
            x='quantity',
            y='price_per_unit',
            color='anomaly_type',
            size='risk_score',
            title="Transaction Anomalies",
            hover_data=['supplier', 'drug_name']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fraud types distribution
        fraud_types = anomaly_df['anomaly_type'].value_counts()
        fig = px.pie(
            values=fraud_types.values,
            names=fraud_types.index,
            title="Fraud Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_ai_decisions(data):
    st.subheader("🤖 AI Decision Engine Analytics")
    
    # Decision history
    decisions = st.session_state.ai_engine.get_decision_history()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Decision timeline
        fig = px.timeline(
            decisions,
            x_start='start_time',
            x_end='end_time',
            y='decision_type',
            color='impact_score',
            title="AI Decision Timeline"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Decision Metrics")
        st.metric("Decisions Made", len(decisions))
        st.metric("Success Rate", f"{decisions['success_rate'].mean():.1%}")
        st.metric("Cost Savings", f"${decisions['cost_impact'].sum():,.2f}")
        st.metric("Time Saved", f"{decisions['time_saved'].sum():.1f} hrs")
    
    # Rules engine performance
    st.markdown("#### ⚙️ Rules Engine Performance")
    rules_data = st.session_state.rules_engine.get_performance_metrics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.bar(
            x=list(rules_data['rule_triggers'].keys()),
            y=list(rules_data['rule_triggers'].values()),
            title="Rule Trigger Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.gauge(
            value=rules_data['overall_efficiency'],
            title="Rules Engine Efficiency",
            range_color="green"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("**Active Rules:**")
        for rule in rules_data['active_rules']:
            st.write(f"• {rule['name']}: {rule['status']}")

def display_analytics(data):
    st.subheader("📈 Supply Chain Analytics & Insights")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Supply chain efficiency
        efficiency_data = data['efficiency_metrics']
        fig = px.line(
            efficiency_data,
            x='date',
            y='efficiency_score',
            title="Supply Chain Efficiency Trend"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cost analysis
        cost_data = data['cost_analysis']
        fig = px.pie(
            values=cost_data['costs'],
            names=cost_data['categories'],
            title="Cost Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Waste reduction
        waste_data = data['waste_metrics']
        fig = px.bar(
            x=waste_data['months'],
            y=waste_data['waste_prevented'],
            title="Waste Prevention by Month"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictive analytics
    st.markdown("#### 🔮 Predictive Analytics")
    predictions = st.session_state.ai_engine.generate_predictions(data)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            predictions['demand_forecast'],
            x='date',
            y='predicted_demand',
            color='drug_category',
            title="Demand Forecast - Next 30 Days"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            predictions['risk_assessment'],
            x='probability',
            y='impact',
            size='priority_score',
            color='risk_type',
            title="Risk Assessment Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_blockchain_audit(data):
    st.subheader("🔗 Blockchain Audit Trail")
    
    # Blockchain status
    blockchain_status = st.session_state.blockchain.get_status()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Blocks", blockchain_status['total_blocks'])
    with col2:
        st.metric("Verified Transactions", f"{blockchain_status['verified_transactions']:,}")
    with col3:
        st.metric("Hash Integrity", f"{blockchain_status['integrity_score']:.1%}")
    with col4:
        st.metric("Network Nodes", blockchain_status['active_nodes'])
    
    # Recent transactions
    st.markdown("#### 📝 Recent Transactions")
    recent_transactions = st.session_state.blockchain.get_recent_transactions()
    
    for tx in recent_transactions[:10]:  # Show last 10 transactions
        with st.expander(f"Transaction {tx['id']} - {tx['type']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Timestamp:** {tx['timestamp']}")
                st.write(f"**Type:** {tx['type']}")
                st.write(f"**Status:** {tx['status']}")
            with col2:
                st.write(f"**Hash:** {tx['hash']}")
                st.write(f"**From:** {tx['from_address']}")
                st.write(f"**To:** {tx['to_address']}")
            st.json(tx['data'])
    
    # Smart contract events
    st.markdown("#### 📋 Smart Contract Events")
    contract_events = st.session_state.blockchain.get_contract_events()
    
    events_df = pd.DataFrame(contract_events)
    if not events_df.empty:
        fig = px.histogram(
            events_df,
            x='event_type',
            title="Smart Contract Event Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        if st.button("📥 Export Audit Trail"):
            csv_data = events_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"blockchain_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
