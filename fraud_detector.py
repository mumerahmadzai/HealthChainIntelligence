import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import hashlib

class FraudDetector:
    def __init__(self):
        self.anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.alert_threshold = 0.7
        self.transaction_patterns = {}
        self.known_fraud_patterns = []
        
    def analyze_transactions(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze transactions for fraud patterns."""
        if not transactions:
            return self._empty_fraud_results()
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(transactions)
        
        # Calculate fraud scores
        fraud_scores = self._calculate_fraud_scores(df)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(df)
        
        # Generate alerts
        alerts = self._generate_alerts(df, fraud_scores, anomalies)
        
        # Create timeline
        timeline = self._create_fraud_timeline(df, fraud_scores)
        
        return {
            'alerts': alerts,
            'anomalies': anomalies,
            'timeline': timeline,
            'fraud_patterns': self._analyze_fraud_patterns(df),
            'summary_stats': self._calculate_summary_stats(df, fraud_scores)
        }
    
    def _calculate_fraud_scores(self, df: pd.DataFrame) -> pd.Series:
        """Calculate fraud risk scores for transactions."""
        fraud_scores = pd.Series(0.0, index=df.index)
        
        # Price anomaly detection
        if 'unit_price' in df.columns and 'drug_id' in df.columns:
            for drug_id in df['drug_id'].unique():
                drug_mask = df['drug_id'] == drug_id
                drug_prices = df.loc[drug_mask, 'unit_price']
                
                if len(drug_prices) > 1:
                    mean_price = drug_prices.mean()
                    std_price = drug_prices.std()
                    
                    if std_price > 0:
                        price_z_scores = np.abs((drug_prices - mean_price) / std_price)
                        fraud_scores.loc[drug_mask] += price_z_scores * 0.3
        
        # Quantity anomaly detection
        if 'quantity' in df.columns:
            quantities = df['quantity']
            q75, q25 = np.percentile(quantities, [75, 25])
            iqr = q75 - q25
            
            # Outliers beyond 1.5 * IQR
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            quantity_anomalies = (quantities < lower_bound) | (quantities > upper_bound)
            fraud_scores.loc[quantity_anomalies] += 0.4
        
        # Temporal pattern analysis
        if 'timestamp' in df.columns and 'supplier_id' in df.columns:
            for supplier_id in df['supplier_id'].unique():
                supplier_mask = df['supplier_id'] == supplier_id
                supplier_transactions = df.loc[supplier_mask].sort_values('timestamp')
                
                if len(supplier_transactions) > 1:
                    # Calculate time differences
                    time_diffs = supplier_transactions['timestamp'].diff().dt.total_seconds() / 3600  # hours
                    
                    # Flag very frequent transactions
                    frequent_mask = time_diffs < 1  # Less than 1 hour apart
                    fraud_scores.loc[supplier_transactions.index[frequent_mask]] += 0.3
        
        # Duplicate transaction detection
        duplicate_columns = ['supplier_id', 'drug_id', 'quantity', 'unit_price']
        if all(col in df.columns for col in duplicate_columns):
            duplicates = df.duplicated(subset=duplicate_columns, keep=False)
            fraud_scores.loc[duplicates] += 0.5
        
        # Supplier reliability factor
        if 'supplier_id' in df.columns:
            # Simulate supplier reliability scores
            supplier_reliability = {}
            for supplier_id in df['supplier_id'].unique():
                supplier_reliability[supplier_id] = np.random.uniform(0.6, 1.0)
            
            for idx, row in df.iterrows():
                reliability = supplier_reliability.get(row['supplier_id'], 0.8)
                if reliability < 0.8:
                    fraud_scores.loc[idx] += (0.8 - reliability) * 2
        
        # Normalize scores to 0-1 range
        if fraud_scores.max() > 0:
            fraud_scores = fraud_scores / fraud_scores.max()
        
        return fraud_scores
    
    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalous transactions using machine learning."""
        if len(df) < 10:
            return pd.DataFrame()
        
        # Prepare features for anomaly detection
        features = []
        feature_names = ['quantity', 'unit_price', 'total_amount']
        
        for col in feature_names:
            if col in df.columns:
                features.append(df[col].fillna(0))
        
        if not features:
            return pd.DataFrame()
        
        feature_matrix = np.column_stack(features)
        
        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Detect anomalies
        anomaly_labels = self.anomaly_model.fit_predict(feature_matrix_scaled)
        anomaly_scores = self.anomaly_model.decision_function(feature_matrix_scaled)
        
        # Create anomaly DataFrame
        anomalies = []
        for idx, (label, score) in enumerate(zip(anomaly_labels, anomaly_scores)):
            if label == -1:  # Anomaly detected
                row = df.iloc[idx]
                anomaly_type = self._classify_anomaly_type(row, df)
                
                anomalies.append({
                    'transaction_id': row.get('id', f'TXN_{idx}'),
                    'timestamp': row.get('timestamp', datetime.now()),
                    'supplier': row.get('supplier_id', 'Unknown'),
                    'drug_name': f"Drug_{row.get('drug_id', 'Unknown')}",
                    'quantity': row.get('quantity', 0),
                    'price_per_unit': row.get('unit_price', 0),
                    'anomaly_type': anomaly_type,
                    'risk_score': abs(score),
                    'anomaly_score': score
                })
        
        return pd.DataFrame(anomalies)
    
    def _classify_anomaly_type(self, transaction: pd.Series, df: pd.DataFrame) -> str:
        """Classify the type of anomaly detected."""
        # Price-based anomalies
        if 'unit_price' in transaction.index:
            drug_prices = df[df['drug_id'] == transaction.get('drug_id', '')]
            if len(drug_prices) > 1:
                mean_price = drug_prices['unit_price'].mean()
                if transaction['unit_price'] > mean_price * 2:
                    return 'Price Manipulation'
                elif transaction['unit_price'] < mean_price * 0.5:
                    return 'Suspicious Pricing'
        
        # Quantity-based anomalies
        if 'quantity' in transaction.index:
            if transaction['quantity'] > 1000:
                return 'Excessive Quantity'
            elif transaction['quantity'] < 1:
                return 'Invalid Quantity'
        
        # Timing-based anomalies
        if 'timestamp' in transaction.index:
            # Check for weekend/holiday transactions
            if transaction['timestamp'].weekday() >= 5:
                return 'Off-hours Activity'
        
        return 'General Anomaly'
    
    def _generate_alerts(self, df: pd.DataFrame, fraud_scores: pd.Series, 
                        anomalies: pd.DataFrame) -> List[Dict]:
        """Generate fraud alerts based on analysis."""
        alerts = []
        
        # High fraud score alerts
        high_risk_transactions = df[fraud_scores > self.alert_threshold]
        
        for idx, row in high_risk_transactions.iterrows():
            alert = {
                'id': f'ALERT_{len(alerts):04d}',
                'timestamp': row.get('timestamp', datetime.now()),
                'type': 'High Risk Transaction',
                'description': f"Transaction {row.get('id', 'Unknown')} has high fraud risk",
                'risk_score': fraud_scores.loc[idx],
                'transaction_id': row.get('id', 'Unknown'),
                'supplier_id': row.get('supplier_id', 'Unknown'),
                'severity': 'High' if fraud_scores.loc[idx] > 0.8 else 'Medium'
            }
            alerts.append(alert)
        
        # Anomaly-based alerts
        for _, anomaly in anomalies.iterrows():
            alert = {
                'id': f'ALERT_{len(alerts):04d}',
                'timestamp': anomaly['timestamp'],
                'type': f'Anomaly Detected: {anomaly["anomaly_type"]}',
                'description': f"Anomalous {anomaly['anomaly_type'].lower()} detected for {anomaly['drug_name']}",
                'risk_score': anomaly['risk_score'],
                'transaction_id': anomaly['transaction_id'],
                'supplier_id': anomaly['supplier'],
                'severity': 'High' if anomaly['risk_score'] > 0.7 else 'Medium'
            }
            alerts.append(alert)
        
        # Pattern-based alerts
        pattern_alerts = self._detect_fraud_patterns(df)
        alerts.extend(pattern_alerts)
        
        # Sort by risk score and timestamp
        alerts.sort(key=lambda x: (x['risk_score'], x['timestamp']), reverse=True)
        
        return alerts[:20]  # Return top 20 alerts
    
    def _detect_fraud_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect complex fraud patterns."""
        pattern_alerts = []
        
        # Round-dollar fraud pattern
        if 'total_amount' in df.columns:
            round_amounts = df[df['total_amount'] % 1 == 0]
            if len(round_amounts) > len(df) * 0.8:  # More than 80% round amounts
                pattern_alerts.append({
                    'id': f'PATTERN_ALERT_{len(pattern_alerts):04d}',
                    'timestamp': datetime.now(),
                    'type': 'Round Dollar Pattern',
                    'description': 'Unusually high frequency of round-dollar transactions detected',
                    'risk_score': 0.6,
                    'transaction_id': 'PATTERN',
                    'supplier_id': 'MULTIPLE',
                    'severity': 'Medium'
                })
        
        # Duplicate shipment pattern
        if 'supplier_id' in df.columns and 'drug_id' in df.columns:
            duplicate_shipments = df.groupby(['supplier_id', 'drug_id', 'quantity']).size()
            suspicious_duplicates = duplicate_shipments[duplicate_shipments > 3]
            
            if len(suspicious_duplicates) > 0:
                pattern_alerts.append({
                    'id': f'PATTERN_ALERT_{len(pattern_alerts):04d}',
                    'timestamp': datetime.now(),
                    'type': 'Duplicate Shipment Pattern',
                    'description': f'Multiple identical shipments detected from {len(suspicious_duplicates)} supplier-drug combinations',
                    'risk_score': 0.7,
                    'transaction_id': 'PATTERN',
                    'supplier_id': 'MULTIPLE',
                    'severity': 'High'
                })
        
        return pattern_alerts
    
    def _create_fraud_timeline(self, df: pd.DataFrame, fraud_scores: pd.Series) -> pd.DataFrame:
        """Create fraud risk timeline."""
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        # Group by day and calculate average fraud score
        df_with_scores = df.copy()
        df_with_scores['fraud_score'] = fraud_scores
        df_with_scores['date'] = df_with_scores['timestamp'].dt.date
        
        timeline = df_with_scores.groupby('date').agg({
            'fraud_score': 'mean',
            'id': 'count'
        }).reset_index()
        
        timeline.columns = ['date', 'fraud_score', 'transaction_count']
        timeline['timestamp'] = pd.to_datetime(timeline['date'])
        
        return timeline
    
    def _analyze_fraud_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall fraud patterns."""
        patterns = {
            'total_transactions': len(df),
            'suspicious_transactions': 0,
            'top_risk_suppliers': [],
            'fraud_by_category': {},
            'temporal_patterns': {}
        }
        
        # Calculate suspicious transactions
        if 'is_fraudulent' in df.columns:
            patterns['suspicious_transactions'] = df['is_fraudulent'].sum()
        
        # Top risk suppliers
        if 'supplier_id' in df.columns:
            supplier_risk = df.groupby('supplier_id').size().sort_values(ascending=False)
            patterns['top_risk_suppliers'] = supplier_risk.head(5).to_dict()
        
        # Fraud by time patterns
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            patterns['temporal_patterns'] = {
                'high_risk_hours': df.groupby('hour').size().sort_values(ascending=False).head(3).to_dict(),
                'high_risk_days': df.groupby('day_of_week').size().sort_values(ascending=False).head(3).to_dict()
            }
        
        return patterns
    
    def _calculate_summary_stats(self, df: pd.DataFrame, fraud_scores: pd.Series) -> Dict[str, Any]:
        """Calculate summary statistics."""
        return {
            'total_transactions_analyzed': len(df),
            'average_fraud_score': fraud_scores.mean(),
            'high_risk_transactions': len(fraud_scores[fraud_scores > self.alert_threshold]),
            'max_fraud_score': fraud_scores.max(),
            'fraud_score_std': fraud_scores.std()
        }
    
    def _empty_fraud_results(self) -> Dict[str, Any]:
        """Return empty fraud analysis results."""
        return {
            'alerts': [],
            'anomalies': pd.DataFrame(),
            'timeline': pd.DataFrame(),
            'fraud_patterns': {},
            'summary_stats': {
                'total_transactions_analyzed': 0,
                'average_fraud_score': 0.0,
                'high_risk_transactions': 0,
                'max_fraud_score': 0.0,
                'fraud_score_std': 0.0
            }
        }
