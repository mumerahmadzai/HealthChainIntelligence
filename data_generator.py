import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Tuple

class DataGenerator:
    def __init__(self):
        self.drug_names = [
            'Amoxicillin', 'Ibuprofen', 'COVID-19 Vaccine', 'Lisinopril', 'Metformin',
            'Atorvastatin', 'Albuterol', 'Acetaminophen', 'Amlodipine', 'Metoprolol',
            'Omeprazole', 'Prednisone', 'Azithromycin', 'Hydrochlorothiazide', 'Gabapentin',
            'Sertraline', 'Furosemide', 'Warfarin', 'Insulin Glargine', 'Levothyroxine',
            'Losartan', 'Simvastatin', 'Montelukast', 'Escitalopram', 'Rosuvastatin'
        ]
        
        self.drug_categories = [
            'Antibiotics', 'Analgesics', 'Vaccines', 'Cardiovascular', 'Diabetes',
            'Respiratory', 'Mental Health', 'Oncology', 'Emergency Medicine'
        ]
        
        self.supplier_names = [
            'MedSupply Corp', 'HealthFirst Distributors', 'PharmaTech Solutions',
            'Global Med Logistics', 'Premier Healthcare Supply', 'MedCore Industries',
            'LifeScience Partners', 'Healthcare Innovations', 'BioMed Distributors',
            'Medical Supply Chain Inc', 'PharmNet Services', 'HealthTech Suppliers'
        ]
        
        self.hospital_names = [
            'General Hospital', 'Medical Center', 'Regional Medical Center',
            'Community Hospital', 'University Hospital', 'Children\'s Hospital',
            'Cancer Center', 'Heart Institute', 'Emergency Medical Center',
            'Specialty Care Center', 'Rehabilitation Hospital', 'Mental Health Center'
        ]
        
        self.locations = [
            'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX',
            'Phoenix, AZ', 'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA',
            'Dallas, TX', 'San Jose, CA', 'Austin, TX', 'Jacksonville, FL',
            'Fort Worth, TX', 'Columbus, OH', 'Charlotte, NC', 'San Francisco, CA'
        ]
    
    def generate_comprehensive_dataset(self, 
                                     num_drugs: int = 50,
                                     num_suppliers: int = 10,
                                     num_hospitals: int = 25,
                                     num_transactions: int = 1000,
                                     days_back: int = 90) -> Dict[str, pd.DataFrame]:
        """Generate a comprehensive dataset for the supply chain simulation."""
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Generate master data
        drugs_df = self.generate_drugs(num_drugs)
        suppliers_df = self.generate_suppliers(num_suppliers)
        hospitals_df = self.generate_hospitals(num_hospitals)
        
        # Generate transactional data
        transactions_df = self.generate_transactions(
            drugs_df, suppliers_df, hospitals_df, num_transactions, days_back
        )
        
        # Generate inventory data
        inventory_df = self.generate_inventory(drugs_df, hospitals_df)
        
        # Generate quality control data
        quality_df = self.generate_quality_control_data(transactions_df)
        
        # Generate performance metrics
        performance_df = self.generate_performance_metrics(suppliers_df, days_back)
        
        return {
            'drugs': drugs_df,
            'suppliers': suppliers_df,
            'hospitals': hospitals_df,
            'transactions': transactions_df,
            'inventory': inventory_df,
            'quality_control': quality_df,
            'performance_metrics': performance_df
        }
    
    def generate_drugs(self, num_drugs: int) -> pd.DataFrame:
        """Generate drug master data."""
        drugs = []
        
        for i in range(num_drugs):
            drug_name = random.choice(self.drug_names)
            strength = random.choice(['5mg', '10mg', '25mg', '50mg', '100mg', '250mg', '500mg'])
            
            drug = {
                'drug_id': f'DRUG_{i:04d}',
                'drug_name': f'{drug_name} {strength}',
                'generic_name': drug_name,
                'brand_name': f'{drug_name} Brand',
                'category': random.choice(self.drug_categories),
                'therapeutic_class': random.choice(['Class A', 'Class B', 'Class C']),
                'unit_cost': round(random.uniform(5.0, 500.0), 2),
                'wholesale_cost': round(random.uniform(3.0, 400.0), 2),
                'retail_price': round(random.uniform(8.0, 600.0), 2),
                'shelf_life_days': random.randint(365, 1825),
                'storage_temp': random.choice(['Room Temperature', 'Refrigerated', 'Frozen']),
                'storage_conditions': random.choice(['Standard', 'Controlled', 'Special']),
                'is_controlled_substance': random.choice([True, False]),
                'is_critical_drug': random.choice([True, False]),
                'minimum_order_quantity': random.randint(10, 100),
                'maximum_order_quantity': random.randint(500, 5000),
                'manufacturer': f'Pharma Company {random.randint(1, 20)}',
                'ndc_number': f'{random.randint(10000, 99999)}-{random.randint(100, 999)}-{random.randint(10, 99)}',
                'lot_size': random.randint(100, 1000),
                'reorder_point': random.randint(20, 200),
                'created_date': datetime.now() - timedelta(days=random.randint(1, 365)),
                'last_updated': datetime.now() - timedelta(days=random.randint(1, 30))
            }
            drugs.append(drug)
        
        return pd.DataFrame(drugs)
    
    def generate_suppliers(self, num_suppliers: int) -> pd.DataFrame:
        """Generate supplier master data."""
        suppliers = []
        
        for i in range(num_suppliers):
            supplier = {
                'supplier_id': f'SUPP_{i:04d}',
                'supplier_name': random.choice(self.supplier_names),
                'supplier_type': random.choice(['Primary', 'Secondary', 'Emergency']),
                'location': random.choice(self.locations),
                'contact_person': f'Contact Person {i+1}',
                'phone': f'({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}',
                'email': f'contact{i+1}@supplier{i+1}.com',
                'reliability_score': round(random.uniform(0.6, 1.0), 3),
                'quality_rating': round(random.uniform(7.0, 10.0), 1),
                'on_time_delivery_rate': round(random.uniform(0.7, 0.98), 3),
                'lead_time_days': random.randint(1, 21),
                'minimum_order_value': random.randint(1000, 10000),
                'payment_terms': random.choice(['Net 30', 'Net 45', 'Net 60', '2/10 Net 30']),
                'discount_rate': round(random.uniform(0.0, 0.15), 3),
                'specialization': random.choice(['General', 'Specialty', 'Vaccines', 'Oncology']),
                'capacity_rating': random.choice(['Low', 'Medium', 'High']),
                'certification_status': random.choice(['Certified', 'Pending', 'Expired']),
                'risk_level': random.choice(['Low', 'Medium', 'High']),
                'contract_start_date': datetime.now() - timedelta(days=random.randint(30, 1095)),
                'contract_end_date': datetime.now() + timedelta(days=random.randint(30, 730)),
                'created_date': datetime.now() - timedelta(days=random.randint(1, 365)),
                'last_updated': datetime.now() - timedelta(days=random.randint(1, 7))
            }
            suppliers.append(supplier)
        
        return pd.DataFrame(suppliers)
    
    def generate_hospitals(self, num_hospitals: int) -> pd.DataFrame:
        """Generate hospital master data."""
        hospitals = []
        
        for i in range(num_hospitals):
            hospital = {
                'hospital_id': f'HOSP_{i:04d}',
                'hospital_name': f'{random.choice(self.hospital_names)} {i+1}',
                'hospital_type': random.choice(['General', 'Specialty', 'Emergency', 'Pediatric', 'Psychiatric']),
                'location': random.choice(self.locations),
                'bed_count': random.randint(50, 800),
                'patient_capacity': random.randint(100, 1500),
                'annual_volume': random.randint(10000, 100000),
                'priority_level': random.choice(['High', 'Medium', 'Low']),
                'service_level': random.choice(['Level 1', 'Level 2', 'Level 3', 'Level 4']),
                'consumption_rate_factor': round(random.uniform(0.5, 2.0), 2),
                'emergency_status': random.choice([True, False]),
                'teaching_hospital': random.choice([True, False]),
                'trauma_center': random.choice([True, False]),
                'specialties': random.choice(['Cardiology', 'Oncology', 'Neurology', 'Pediatrics', 'General']),
                'pharmacy_director': f'Director {i+1}',
                'procurement_contact': f'Procurement {i+1}',
                'phone': f'({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}',
                'email': f'pharmacy{i+1}@hospital{i+1}.org',
                'license_number': f'LIC{random.randint(10000, 99999)}',
                'accreditation_status': random.choice(['Accredited', 'Provisional', 'Under Review']),
                'created_date': datetime.now() - timedelta(days=random.randint(1, 365)),
                'last_updated': datetime.now() - timedelta(days=random.randint(1, 7))
            }
            hospitals.append(hospital)
        
        return pd.DataFrame(hospitals)
    
    def generate_transactions(self, drugs_df: pd.DataFrame, suppliers_df: pd.DataFrame,
                            hospitals_df: pd.DataFrame, num_transactions: int, days_back: int) -> pd.DataFrame:
        """Generate transaction data."""
        transactions = []
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for i in range(num_transactions):
            # Random transaction date
            transaction_date = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            # Select random entities
            drug = drugs_df.sample(1).iloc[0]
            supplier = suppliers_df.sample(1).iloc[0]
            hospital = hospitals_df.sample(1).iloc[0]
            
            # Determine if this is a fraudulent transaction (5% chance)
            is_fraudulent = random.random() < 0.05
            
            # Base quantity
            base_quantity = random.randint(10, 500)
            
            # Introduce fraud patterns
            if is_fraudulent:
                if random.random() < 0.3:  # Excessive quantity
                    base_quantity *= random.randint(3, 10)
                elif random.random() < 0.3:  # Duplicate pattern
                    # Will be handled by creating similar transactions
                    pass
            
            # Calculate pricing
            unit_price = drug['unit_cost'] * random.uniform(0.8, 1.2)
            if is_fraudulent and random.random() < 0.4:  # Price manipulation
                unit_price *= random.uniform(1.5, 3.0)
            
            transaction = {
                'transaction_id': f'TXN_{i:08d}',
                'transaction_date': transaction_date,
                'transaction_type': random.choice(['Purchase', 'Transfer', 'Return', 'Adjustment']),
                'drug_id': drug['drug_id'],
                'supplier_id': supplier['supplier_id'],
                'hospital_id': hospital['hospital_id'],
                'quantity': base_quantity,
                'unit_price': round(unit_price, 2),
                'total_amount': round(base_quantity * unit_price, 2),
                'currency': 'USD',
                'batch_number': f'BATCH_{random.randint(100000, 999999)}',
                'lot_number': f'LOT_{random.randint(10000, 99999)}',
                'expiry_date': transaction_date + timedelta(days=drug['shelf_life_days']),
                'manufacture_date': transaction_date - timedelta(days=random.randint(1, 180)),
                'po_number': f'PO_{random.randint(100000, 999999)}',
                'invoice_number': f'INV_{random.randint(100000, 999999)}',
                'delivery_date': transaction_date + timedelta(days=random.randint(1, supplier['lead_time_days'])),
                'payment_terms': supplier['payment_terms'],
                'discount_applied': round(random.uniform(0.0, supplier['discount_rate']), 3),
                'tax_amount': round(base_quantity * unit_price * 0.08, 2),  # 8% tax
                'shipping_cost': round(random.uniform(10.0, 100.0), 2),
                'transaction_status': random.choice(['Completed', 'Pending', 'Cancelled', 'On Hold']),
                'quality_check_status': random.choice(['Passed', 'Pending', 'Failed']),
                'temperature_log': random.choice(['Within Range', 'Deviation Detected', 'Not Monitored']),
                'is_emergency_order': random.choice([True, False]),
                'priority_level': random.choice(['Standard', 'High', 'Emergency']),
                'approved_by': f'User_{random.randint(1, 20)}',
                'created_by': f'System_User_{random.randint(1, 10)}',
                'is_fraudulent': is_fraudulent,
                'fraud_score': random.uniform(0.1, 0.9) if is_fraudulent else random.uniform(0.0, 0.3),
                'verification_status': random.choice(['Verified', 'Pending', 'Rejected']),
                'audit_trail': f'Created: {transaction_date}, Modified: {transaction_date}',
                'notes': 'Auto-generated transaction' if not is_fraudulent else 'Flagged for review',
                'created_date': transaction_date,
                'last_updated': transaction_date + timedelta(minutes=random.randint(1, 60))
            }
            
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def generate_inventory(self, drugs_df: pd.DataFrame, hospitals_df: pd.DataFrame) -> pd.DataFrame:
        """Generate current inventory data."""
        inventory = []
        
        for _, hospital in hospitals_df.iterrows():
            # Each hospital has inventory for a subset of drugs
            num_drugs_in_hospital = random.randint(20, len(drugs_df))
            hospital_drugs = drugs_df.sample(num_drugs_in_hospital)
            
            for _, drug in hospital_drugs.iterrows():
                current_stock = random.randint(0, 1000)
                max_stock = random.randint(500, 2000)
                min_stock = random.randint(10, 100)
                
                # Determine stock status
                stock_ratio = current_stock / max_stock if max_stock > 0 else 0
                if stock_ratio < 0.1:
                    stock_status = 'Critical'
                elif stock_ratio < 0.3:
                    stock_status = 'Low'
                elif stock_ratio > 0.8:
                    stock_status = 'Overstocked'
                else:
                    stock_status = 'Normal'
                
                inventory_item = {
                    'inventory_id': f'INV_{len(inventory):08d}',
                    'hospital_id': hospital['hospital_id'],
                    'drug_id': drug['drug_id'],
                    'current_stock': current_stock,
                    'available_stock': current_stock - random.randint(0, min(current_stock, 50)),
                    'reserved_stock': random.randint(0, min(current_stock, 100)),
                    'min_stock_level': min_stock,
                    'max_stock_level': max_stock,
                    'reorder_point': random.randint(min_stock, min_stock * 2),
                    'stock_status': stock_status,
                    'average_daily_usage': round(random.uniform(1.0, 20.0), 2),
                    'days_of_supply': round(current_stock / max(1, random.uniform(1.0, 20.0)), 1),
                    'last_restock_date': datetime.now() - timedelta(days=random.randint(1, 60)),
                    'next_restock_date': datetime.now() + timedelta(days=random.randint(1, 30)),
                    'supplier_id': random.choice(hospitals_df['hospital_id'].tolist()),  # Primary supplier
                    'storage_location': f'Ward {random.choice(["A", "B", "C"])}-{random.randint(1, 20)}',
                    'storage_condition': drug['storage_temp'],
                    'batch_numbers': [f'BATCH_{random.randint(100000, 999999)}' for _ in range(random.randint(1, 3))],
                    'expiry_dates': [datetime.now() + timedelta(days=random.randint(30, 365)) for _ in range(random.randint(1, 3))],
                    'cost_per_unit': drug['unit_cost'] * random.uniform(0.9, 1.1),
                    'total_value': current_stock * drug['unit_cost'] * random.uniform(0.9, 1.1),
                    'turnover_ratio': round(random.uniform(2.0, 12.0), 2),
                    'abc_classification': random.choice(['A', 'B', 'C']),
                    'is_consignment': random.choice([True, False]),
                    'quality_status': random.choice(['Good', 'Quarantine', 'Expired']),
                    'temperature_monitored': random.choice([True, False]),
                    'last_audit_date': datetime.now() - timedelta(days=random.randint(1, 90)),
                    'audit_status': random.choice(['Passed', 'Failed', 'Pending']),
                    'created_date': datetime.now() - timedelta(days=random.randint(1, 180)),
                    'last_updated': datetime.now() - timedelta(hours=random.randint(1, 48))
                }
                
                inventory.append(inventory_item)
        
        return pd.DataFrame(inventory)
    
    def generate_quality_control_data(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate quality control data."""
        quality_checks = []
        
        # Generate QC data for a subset of transactions
        qc_transactions = transactions_df.sample(min(200, len(transactions_df)))
        
        for _, transaction in qc_transactions.iterrows():
            quality_check = {
                'qc_id': f'QC_{len(quality_checks):08d}',
                'transaction_id': transaction['transaction_id'],
                'batch_number': transaction['batch_number'],
                'drug_id': transaction['drug_id'],
                'supplier_id': transaction['supplier_id'],
                'qc_date': transaction['transaction_date'] + timedelta(hours=random.randint(1, 48)),
                'qc_type': random.choice(['Incoming Inspection', 'Periodic Review', 'Batch Testing', 'Temperature Check']),
                'test_parameters': random.choice(['Visual Inspection', 'Chemical Analysis', 'Potency Test', 'Contamination Check']),
                'test_results': random.choice(['Pass', 'Fail', 'Conditional Pass']),
                'quality_score': round(random.uniform(7.0, 10.0), 1),
                'temperature_reading': round(random.uniform(2.0, 8.0), 1) if transaction.get('storage_temp') == 'Refrigerated' else None,
                'humidity_reading': round(random.uniform(30.0, 70.0), 1),
                'packaging_condition': random.choice(['Good', 'Minor Damage', 'Major Damage']),
                'labeling_accuracy': random.choice(['Correct', 'Minor Error', 'Major Error']),
                'documentation_complete': random.choice([True, False]),
                'expiry_date_verified': random.choice([True, False]),
                'batch_size_verified': random.choice([True, False]),
                'certificate_of_analysis': random.choice([True, False]),
                'deviation_noted': random.choice([True, False]),
                'corrective_action_required': random.choice([True, False]),
                'qc_technician': f'QC_Tech_{random.randint(1, 10)}',
                'supervisor_approval': f'QC_Supervisor_{random.randint(1, 5)}',
                'notes': 'Standard quality control check completed',
                'follow_up_required': random.choice([True, False]),
                'follow_up_date': datetime.now() + timedelta(days=random.randint(1, 30)) if random.choice([True, False]) else None,
                'created_date': transaction['transaction_date'] + timedelta(hours=random.randint(1, 48)),
                'last_updated': datetime.now()
            }
            
            quality_checks.append(quality_check)
        
        return pd.DataFrame(quality_checks)
    
    def generate_performance_metrics(self, suppliers_df: pd.DataFrame, days_back: int) -> pd.DataFrame:
        """Generate supplier performance metrics over time."""
        performance_data = []
        
        # Generate daily performance metrics for each supplier
        for _, supplier in suppliers_df.iterrows():
            for day in range(days_back):
                metric_date = datetime.now() - timedelta(days=day)
                
                # Base performance with some variation
                base_reliability = supplier['reliability_score']
                daily_variation = random.uniform(-0.1, 0.1)
                
                performance = {
                    'performance_id': f'PERF_{len(performance_data):08d}',
                    'supplier_id': supplier['supplier_id'],
                    'metric_date': metric_date,
                    'on_time_delivery_rate': max(0, min(1, base_reliability + daily_variation)),
                    'quality_score': round(random.uniform(7.0, 10.0), 1),
                    'order_accuracy': round(random.uniform(0.85, 0.99), 3),
                    'response_time_hours': round(random.uniform(1.0, 48.0), 1),
                    'fill_rate': round(random.uniform(0.8, 1.0), 3),
                    'cost_competitiveness': round(random.uniform(0.7, 1.0), 3),
                    'communication_rating': round(random.uniform(6.0, 10.0), 1),
                    'compliance_score': round(random.uniform(8.0, 10.0), 1),
                    'orders_processed': random.randint(1, 20),
                    'orders_on_time': random.randint(0, 20),
                    'orders_delayed': random.randint(0, 5),
                    'average_delay_hours': round(random.uniform(0.0, 72.0), 1),
                    'defect_rate': round(random.uniform(0.0, 0.05), 4),
                    'return_rate': round(random.uniform(0.0, 0.03), 4),
                    'customer_satisfaction': round(random.uniform(7.0, 10.0), 1),
                    'contract_compliance': round(random.uniform(0.9, 1.0), 3),
                    'innovation_score': round(random.uniform(5.0, 10.0), 1),
                    'sustainability_score': round(random.uniform(6.0, 10.0), 1),
                    'created_date': metric_date,
                    'last_updated': metric_date + timedelta(hours=1)
                }
                
                performance_data.append(performance)
        
        return pd.DataFrame(performance_data)
    
    def introduce_anomalies(self, df: pd.DataFrame, anomaly_rate: float = 0.05) -> pd.DataFrame:
        """Introduce anomalies into the dataset for testing fraud detection."""
        anomaly_count = int(len(df) * anomaly_rate)
        anomaly_indices = np.random.choice(df.index, anomaly_count, replace=False)
        
        df_with_anomalies = df.copy()
        
        for idx in anomaly_indices:
            if 'quantity' in df.columns:
                if np.random.random() < 0.5:
                    df_with_anomalies.loc[idx, 'quantity'] *= np.random.randint(5, 20)
            
            if 'unit_price' in df.columns:
                if np.random.random() < 0.5:
                    df_with_anomalies.loc[idx, 'unit_price'] *= np.random.uniform(2.0, 5.0)
            
            if 'is_fraudulent' in df.columns:
                df_with_anomalies.loc[idx, 'is_fraudulent'] = True
        
        return df_with_anomalies
    
    def generate_time_series_data(self, metric_name: str, days: int, trend: str = 'stable') -> pd.DataFrame:
        """Generate time series data for various metrics."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
        
        if trend == 'increasing':
            trend_component = np.linspace(0, 0.3, days)
        elif trend == 'decreasing':
            trend_component = np.linspace(0.3, 0, days)
        else:
            trend_component = np.zeros(days)
        
        # Seasonal component
        seasonal_component = 0.1 * np.sin(2 * np.pi * np.arange(days) / 7)  # Weekly seasonality
        
        # Random noise
        noise = np.random.normal(0, 0.05, days)
        
        # Base value
        base_value = 0.8
        
        values = base_value + trend_component + seasonal_component + noise
        values = np.clip(values, 0, 1)  # Keep values between 0 and 1
        
        return pd.DataFrame({
            'date': dates,
            'metric_name': metric_name,
            'value': values
        })
