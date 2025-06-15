# AI-Driven Healthcare Supply Chain Simulation

## Overview

This project is a comprehensive simulation system for healthcare supply chain management, built with Python and Streamlit. It demonstrates intelligent contract automation using AI models, blockchain technology, and real-time fraud detection for pharmaceutical supply chains. The system is designed as a research prototype for DBA studies on supply chain optimization in healthcare.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Interactive dashboard for simulation control and visualization
- **Real-time Data Visualization**: Uses Plotly for charts and graphs
- **Responsive UI**: Wide layout with expandable sidebar for parameter controls
- **Multi-tab Interface**: Separate views for different simulation aspects

### Backend Architecture
- **Modular Design**: Separate modules for different functionalities
- **Event-Driven Simulation**: Time-based simulation engine with configurable parameters
- **AI Decision Engine**: Machine learning models for fraud detection and demand prediction
- **Rules Engine**: Configurable business rules for supply chain automation
- **Blockchain Simulator**: Simulated distributed ledger for transaction verification

### Core Components
1. **Supply Chain Simulator** (`supply_chain_simulator.py`): Main simulation engine
2. **AI Models** (`ai_models.py`): Machine learning components for decision making
3. **Fraud Detector** (`fraud_detector.py`): Anomaly detection and fraud prevention
4. **Rules Engine** (`rules_engine.py`): Business logic automation
5. **Blockchain Simulator** (`blockchain_simulator.py`): Distributed ledger simulation
6. **Data Generator** (`data_generator.py`): Synthetic data creation for testing

## Key Components

### Supply Chain Simulator
- Manages drugs, suppliers, hospitals, and transactions
- Generates realistic pharmaceutical inventory data
- Simulates supply chain events over configurable time periods
- Handles drug categories, pricing, expiration dates, and storage requirements

### AI Decision Engine
- **Fraud Detection**: Uses Isolation Forest for anomaly detection
- **Demand Prediction**: Linear regression for forecasting
- **Inventory Management**: Automated restock recommendations
- **Configurable Parameters**: Adjustable sensitivity and thresholds

### Blockchain Integration
- **Transaction Types**: Drug procurement, inventory transfers, quality verification
- **Smart Contracts**: Automated execution of supply chain rules
- **Immutable Records**: Tamper-proof transaction history
- **Consensus Mechanism**: Simulated blockchain validation

### Rules Engine
- **Flexible Rule Definition**: Configurable business logic
- **Priority System**: Four-level priority classification
- **Execution History**: Complete audit trail of rule executions
- **Performance Metrics**: Real-time monitoring of rule effectiveness

## Data Flow

1. **Initialization**: System generates synthetic data for drugs, suppliers, and hospitals
2. **Simulation Loop**: Processes daily supply chain events
3. **AI Analysis**: Models analyze transactions for fraud and demand patterns
4. **Rule Execution**: Business rules trigger automated responses
5. **Blockchain Recording**: All transactions are recorded on simulated blockchain
6. **Visualization**: Real-time updates to dashboard metrics and charts

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive data visualization
- **Scikit-learn**: Machine learning models
- **Joblib**: Model persistence and parallel processing

### Development Tools
- **Python 3.11**: Core runtime environment
- **Nix**: Package management and environment isolation
- **UV**: Fast Python package installer

## Deployment Strategy

### Replit Configuration
- **Target**: Autoscale deployment for dynamic resource allocation
- **Port**: 5000 (configured for Streamlit server)
- **Workflow**: Parallel execution with automated startup
- **Environment**: Python 3.11 with Nix package management

### Scalability Considerations
- Modular architecture allows for easy component scaling
- Stateless design enables horizontal scaling
- In-memory data storage suitable for demonstration purposes
- Future database integration supported through modular design

### Configuration Management
- Environment-specific settings in `.streamlit/config.toml`
- Dependency management through `pyproject.toml`
- Replit-specific configuration in `.replit` file

## Changelog

```
Changelog:
- June 15, 2025. Initial setup
- June 15, 2025. Completed AI-driven healthcare supply chain simulation with working:
  * Machine learning fraud detection using IsolationForest and RandomForest
  * Demand prediction with RandomForestRegressor
  * Automated restock recommendations with configurable thresholds
  * Rules engine for supply chain automation with 8 default rules
  * Blockchain simulator with smart contract events
  * Interactive Streamlit dashboard with 5 main tabs
  * Real-time inventory tracking and analytics
  * All technical issues resolved and application fully functional
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```

## Technical Notes

### AI Model Architecture
The system uses ensemble methods combining:
- Isolation Forest for anomaly detection
- Random Forest for classification tasks
- Linear Regression for demand forecasting
- Standard scaling for feature normalization

### Blockchain Simulation
- Simulates proof-of-work consensus
- Implements smart contract events
- Maintains transaction integrity
- Provides audit trail capabilities

### Data Generation
- Realistic pharmaceutical data
- Configurable dataset sizes
- Time-series transaction patterns
- Multi-dimensional fraud scenarios

The system is designed for research and demonstration purposes, showcasing how AI and blockchain technologies can be integrated into healthcare supply chain management.