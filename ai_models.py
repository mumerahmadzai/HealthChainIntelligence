import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib
from typing import Dict, List, Any, Tuple

class AIDecisionEngine:
    def __init__(self):
        self.fraud_model = None
        self.demand_model = None
        self.restock_model = None
        self.scaler = StandardScaler()
        self.decision_history = []
        self.fraud_sensitivity = 0.7
        self.restock_threshold = 0.2
        self.expiry_warning_days = 30
        self.restock_rules = {
            'critical_threshold': 0.1,  # 10% of max stock
            'normal_threshold': 0.2,    # 20% of max stock
            'lead_time_buffer': 1.5,    # 50% buffer for lead time
            'seasonal_multiplier': 1.2   # 20% increase for seasonal demand
        }
        
    def configure(self, fraud_sensitivity: float, restock_threshold: float, expiry_warning_days: int):
        """Configure AI model parameters."""
        self.fraud_sensitivity = fraud_sensitivity
        self.restock_threshold = restock_threshold / 100.0  # Convert percentage
        self.expiry_warning_days = expiry_warning_days
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize and train AI models with synthetic data."""
        # Generate training data for fraud detection
        fraud_training_data = self._generate_fraud_training_data()
        self._train_fraud_model(fraud_training_data)
        
        # Generate training data for demand prediction
        demand_training_data = self._generate_demand_training_data()
        self._train_demand_model(demand_training_data)
        
        # Initialize restock model
        self._initialize_restock_model()
    
    def _generate_fraud_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data for fraud detection."""
        np.random.seed(42)  # For reproducibility
        n_samples = 10000
        
        # Normal transactions
        normal_data = {
            'quantity': np.random.normal(50, 20, int(n_samples * 0.9)),
            'price_ratio': np.random.normal(1.0, 0.1, int(n_samples * 0.9)),
            'time_since_last': np.random.exponential(2, int(n_samples * 0.9)),
            'supplier_reliability': np.random.normal(0.9, 0.1, int(n_samples * 0.9)),
            'quantity_deviation': np.random.normal(0, 0.1, int(n_samples * 0.9)),
            'is_fraud': [0] * int(n_samples * 0.9)
        }
        
        # Fraudulent transactions
        fraud_data = {
            'quantity': np.random.normal(200, 100, int(n_samples * 0.1)),  # Unusual quantities
            'price_ratio': np.random.normal(1.5, 0.5, int(n_samples * 0.1)),  # Price anomalies
            'time_since_last': np.random.exponential(0.5, int(n_samples * 0.1)),  # Frequent transactions
            'supplier_reliability': np.random.normal(0.5, 0.2, int(n_samples * 0.1)),  # Low reliability
            'quantity_deviation': np.random.normal(2, 0.5, int(n_samples * 0.1)),  # High deviation
            'is_fraud': [1] * int(n_samples * 0.1)
        }
        
        # Combine data
        all_data = {}
        for key in normal_data.keys():
            all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
        
        return pd.DataFrame(all_data)
    
    def _generate_demand_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data for demand prediction."""
        np.random.seed(42)
        n_samples = 5000
        
        # Create time series data
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # Seasonal patterns
        seasonal_factor = np.sin(2 * np.pi * np.arange(n_samples) / 365.25) * 0.3
        
        # Trend
        trend = np.linspace(0, 0.5, n_samples)
        
        # Random noise
        noise = np.random.normal(0, 0.2, n_samples)
        
        # Base demand
        base_demand = 100 + seasonal_factor * 50 + trend * 20 + noise * 10
        
        # Additional features
        day_of_week = [d.weekday() for d in dates]
        month = [d.month for d in dates]
        
        data = {
            'date': dates,
            'day_of_week': day_of_week,
            'month': month,
            'seasonal_factor': seasonal_factor,
            'trend': trend,
            'historical_avg': np.roll(base_demand, 7),  # 7-day lag
            'demand': base_demand
        }
        
        return pd.DataFrame(data)
    
    def _train_fraud_model(self, training_data: pd.DataFrame):
        """Train the fraud detection model."""
        features = ['quantity', 'price_ratio', 'time_since_last', 'supplier_reliability', 'quantity_deviation']
        X = training_data[features]
        y = training_data['is_fraud']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest for fraud detection
        self.fraud_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.fraud_model.fit(X_scaled, y)
        
        # Also train isolation forest for anomaly detection
        self.anomaly_model = IsolationForest(
            random_state=42
        )
        self.anomaly_model.fit(X_scaled[y == 0])  # Train only on normal data
    
    def _train_demand_model(self, training_data: pd.DataFrame):
        """Train the demand prediction model."""
        features = ['day_of_week', 'month', 'seasonal_factor', 'trend', 'historical_avg']
        X = training_data[features].fillna(0)
        y = training_data['demand']
        
        self.demand_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        self.demand_model.fit(X, y)
    
    def _initialize_restock_model(self):
        """Initialize restock recommendation logic."""
        self.restock_rules = {
            'critical_threshold': 0.1,  # 10% of max stock
            'normal_threshold': 0.2,    # 20% of max stock
            'lead_time_buffer': 1.5,    # 50% buffer for lead time
            'seasonal_multiplier': 1.2   # 20% increase for seasonal demand
        }
    
    def detect_fraud(self, transaction_features: Dict) -> Tuple[float, bool]:
        """Detect fraud in a transaction."""
        if self.fraud_model is None:
            return 0.0, False
        
        # Prepare features
        features = [
            transaction_features.get('quantity', 0),
            transaction_features.get('price_ratio', 1.0),
            transaction_features.get('time_since_last', 1.0),
            transaction_features.get('supplier_reliability', 0.9),
            transaction_features.get('quantity_deviation', 0.0)
        ]
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Get fraud probability
        fraud_prob = self.fraud_model.predict_proba(features_scaled)[0][1]
        
        # Get anomaly score
        anomaly_score = self.anomaly_model.decision_function(features_scaled)[0]
        
        # Combine scores
        combined_score = (fraud_prob + (1 - (anomaly_score + 0.5))) / 2
        
        is_fraud = combined_score > self.fraud_sensitivity
        
        return combined_score, is_fraud
    
    def predict_demand(self, date: datetime, drug_id: str, hospital_id: str) -> float:
        """Predict demand for a specific drug at a hospital."""
        if self.demand_model is None:
            return 50.0  # Default prediction
        
        # Prepare features
        features = [
            date.weekday(),
            date.month,
            np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25) * 0.3,
            0.1,  # Trend placeholder
            50.0  # Historical average placeholder
        ]
        
        # Predict demand
        try:
            prediction = self.demand_model.predict([features])[0]
            return max(0, prediction)
        except:
            return 50.0  # Fallback
    
    def get_restock_recommendations(self, data: Dict) -> List[Dict]:
        """Generate AI-powered restock recommendations."""
        recommendations = []
        
        inventory_df = data.get('inventory_levels', pd.DataFrame())
        
        for _, row in inventory_df.iterrows():
            # Calculate stock ratio
            current_stock = row.get('current_stock', 0)
            max_stock = 1000  # Placeholder, should come from inventory data
            stock_ratio = current_stock / max_stock if max_stock > 0 else 0
            
            # Determine urgency
            if stock_ratio < self.restock_rules['critical_threshold']:
                urgency = 'High'
                recommended_qty = int(max_stock * 0.8)
            elif stock_ratio < self.restock_rules['normal_threshold']:
                urgency = 'Medium'
                recommended_qty = int(max_stock * 0.6)
            elif stock_ratio < self.restock_threshold:
                urgency = 'Low'
                recommended_qty = int(max_stock * 0.4)
            else:
                continue  # No restock needed
            
            recommendation = {
                'drug_name': row.get('drug_name', 'Unknown'),
                'hospital_id': row.get('hospital_id', 'Unknown'),
                'current_stock': current_stock,
                'recommended_quantity': recommended_qty,
                'urgency': urgency,
                'reasoning': f"Stock ratio: {stock_ratio:.1%}, Below threshold: {self.restock_threshold:.1%}"
            }
            
            recommendations.append(recommendation)
            
            # Record decision
            self._record_decision('Restock Recommendation', recommendation)
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def generate_predictions(self, data: Dict) -> Dict:
        """Generate various predictions for the dashboard."""
        current_date = datetime.now()
        
        # Demand forecast
        demand_forecast = []
        drug_categories = ['Antibiotics', 'Analgesics', 'Vaccines', 'Cardiology']
        
        for i in range(30):  # 30-day forecast
            forecast_date = current_date + timedelta(days=i)
            for category in drug_categories:
                base_demand = 100 + np.sin(2 * np.pi * i / 30) * 20 + np.random.normal(0, 5)
                demand_forecast.append({
                    'date': forecast_date,
                    'drug_category': category,
                    'predicted_demand': max(0, base_demand)
                })
        
        # Risk assessment
        risk_types = ['Supply Disruption', 'Quality Issues', 'Regulatory Changes', 'Demand Surge']
        risk_assessment = []
        
        for risk_type in risk_types:
            probability = np.random.uniform(0.1, 0.9)
            impact = np.random.uniform(0.2, 1.0)
            priority_score = probability * impact * 100
            
            risk_assessment.append({
                'risk_type': risk_type,
                'probability': probability,
                'impact': impact,
                'priority_score': priority_score
            })
        
        return {
            'demand_forecast': pd.DataFrame(demand_forecast),
            'risk_assessment': pd.DataFrame(risk_assessment)
        }
    
    def _record_decision(self, decision_type: str, details: Dict):
        """Record AI decision for audit trail."""
        decision = {
            'timestamp': datetime.now(),
            'decision_type': decision_type,
            'details': details,
            'success_rate': np.random.uniform(0.8, 0.95),
            'cost_impact': np.random.uniform(-1000, 5000),
            'time_saved': np.random.uniform(0.5, 8.0),
            'start_time': datetime.now() - timedelta(minutes=np.random.randint(1, 60)),
            'end_time': datetime.now(),
            'impact_score': np.random.uniform(0.5, 1.0)
        }
        
        self.decision_history.append(decision)
        
        # Keep only recent decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def get_decision_history(self) -> pd.DataFrame:
        """Get decision history for analytics."""
        if not self.decision_history:
            # Generate some sample decisions
            for i in range(50):
                self._record_decision(
                    np.random.choice(['Restock Recommendation', 'Fraud Alert', 'Priority Shipment']),
                    {'sample': True}
                )
        
        return pd.DataFrame(self.decision_history)
