import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any

class SupplyChainSimulator:
    def __init__(self):
        self.drugs = []
        self.suppliers = []
        self.hospitals = []
        self.transactions = []
        self.inventory = {}
        self.simulation_start = datetime.now()
        self.current_day = 0
        
    def initialize(self, num_drugs: int, num_suppliers: int, num_hospitals: int, days: int):
        """Initialize the simulation with specified parameters."""
        self.simulation_days = days
        self._generate_drugs(num_drugs)
        self._generate_suppliers(num_suppliers)
        self._generate_hospitals(num_hospitals)
        self._generate_initial_inventory()
        self._simulate_transactions()
        
    def _generate_drugs(self, num_drugs: int):
        """Generate drug catalog with realistic healthcare drugs."""
        drug_categories = ['Antibiotics', 'Analgesics', 'Vaccines', 'Cardiology', 'Oncology', 'Diabetes', 'Respiratory']
        drug_names = [
            'Amoxicillin', 'Ibuprofen', 'COVID-19 Vaccine', 'Lisinopril', 'Metformin',
            'Atorvastatin', 'Albuterol', 'Acetaminophen', 'Amlodipine', 'Metoprolol',
            'Omeprazole', 'Prednisone', 'Azithromycin', 'Hydrochlorothiazide', 'Gabapentin',
            'Sertraline', 'Furosemide', 'Warfarin', 'Insulin', 'Levothyroxine'
        ]
        
        self.drugs = []
        for i in range(num_drugs):
            drug = {
                'id': f'DRUG_{i:04d}',
                'name': random.choice(drug_names) + f' {random.randint(10, 500)}mg',
                'category': random.choice(drug_categories),
                'unit_cost': round(random.uniform(5.0, 500.0), 2),
                'shelf_life_days': random.randint(365, 1825),  # 1-5 years
                'critical_drug': random.choice([True, False]),
                'storage_temp': random.choice(['Room Temperature', 'Refrigerated', 'Frozen']),
                'manufacturer': f'Pharma_{random.randint(1, 20)}'
            }
            self.drugs.append(drug)
    
    def _generate_suppliers(self, num_suppliers: int):
        """Generate supplier network."""
        self.suppliers = []
        for i in range(num_suppliers):
            supplier = {
                'id': f'SUPP_{i:04d}',
                'name': f'Healthcare Supplier {i+1}',
                'reliability_score': round(random.uniform(0.7, 1.0), 2),
                'location': random.choice(['New York', 'California', 'Texas', 'Florida', 'Illinois']),
                'specialization': random.choice(['General', 'Specialty', 'Emergency', 'Vaccines']),
                'capacity': random.randint(1000, 10000),
                'lead_time_days': random.randint(1, 14)
            }
            self.suppliers.append(supplier)
    
    def _generate_hospitals(self, num_hospitals: int):
        """Generate hospital network."""
        self.hospitals = []
        for i in range(num_hospitals):
            hospital = {
                'id': f'HOSP_{i:04d}',
                'name': f'Medical Center {i+1}',
                'type': random.choice(['General', 'Specialty', 'Emergency', 'Pediatric']),
                'bed_count': random.randint(50, 800),
                'location': random.choice(['Urban', 'Suburban', 'Rural']),
                'priority_level': random.choice(['High', 'Medium', 'Low']),
                'consumption_rate': round(random.uniform(0.1, 2.0), 2)
            }
            self.hospitals.append(hospital)
    
    def _generate_initial_inventory(self):
        """Generate initial inventory levels."""
        self.inventory = {}
        for drug in self.drugs:
            for hospital in self.hospitals:
                key = f"{hospital['id']}_{drug['id']}"
                self.inventory[key] = {
                    'hospital_id': hospital['id'],
                    'drug_id': drug['id'],
                    'current_stock': random.randint(10, 500),
                    'min_stock': random.randint(5, 50),
                    'max_stock': random.randint(200, 1000),
                    'expiry_date': self.simulation_start + timedelta(
                        days=random.randint(30, drug['shelf_life_days'])
                    ),
                    'batch_number': f"BATCH_{random.randint(10000, 99999)}",
                    'last_updated': self.simulation_start
                }
    
    def _simulate_transactions(self):
        """Simulate supply chain transactions."""
        self.transactions = []
        
        for day in range(self.simulation_days):
            current_date = self.simulation_start + timedelta(days=day)
            
            # Generate daily transactions
            num_transactions = random.randint(5, 25)
            
            for _ in range(num_transactions):
                transaction = self._generate_transaction(current_date)
                self.transactions.append(transaction)
                
                # Update inventory
                self._update_inventory(transaction)
    
    def _generate_transaction(self, date: datetime) -> Dict:
        """Generate a single transaction."""
        transaction_type = random.choice(['Purchase', 'Transfer', 'Consumption', 'Disposal'])
        supplier = random.choice(self.suppliers)
        hospital = random.choice(self.hospitals)
        drug = random.choice(self.drugs)
        
        # Introduce some fraudulent patterns
        is_fraudulent = random.random() < 0.05  # 5% fraud rate
        
        base_quantity = random.randint(1, 100)
        if is_fraudulent:
            # Fraudulent patterns
            if random.random() < 0.3:  # Duplicate shipment
                base_quantity *= 2
            elif random.random() < 0.3:  # Unusual quantity
                base_quantity *= random.randint(5, 20)
        
        transaction = {
            'id': f'TXN_{len(self.transactions):08d}',
            'timestamp': date,
            'type': transaction_type,
            'supplier_id': supplier['id'],
            'hospital_id': hospital['id'],
            'drug_id': drug['id'],
            'quantity': base_quantity,
            'unit_price': drug['unit_cost'] * random.uniform(0.8, 1.2),
            'total_amount': 0,  # Will be calculated
            'batch_number': f"BATCH_{random.randint(10000, 99999)}",
            'expiry_date': date + timedelta(days=drug['shelf_life_days']),
            'is_fraudulent': is_fraudulent,
            'verification_status': 'Pending'
        }
        
        transaction['total_amount'] = transaction['quantity'] * transaction['unit_price']
        
        return transaction
    
    def _update_inventory(self, transaction: Dict):
        """Update inventory based on transaction."""
        key = f"{transaction['hospital_id']}_{transaction['drug_id']}"
        
        if key in self.inventory:
            if transaction['type'] == 'Purchase':
                self.inventory[key]['current_stock'] += transaction['quantity']
            elif transaction['type'] == 'Consumption':
                self.inventory[key]['current_stock'] = max(0, 
                    self.inventory[key]['current_stock'] - transaction['quantity'])
            elif transaction['type'] == 'Disposal':
                self.inventory[key]['current_stock'] = max(0,
                    self.inventory[key]['current_stock'] - transaction['quantity'])
            
            self.inventory[key]['last_updated'] = transaction['timestamp']
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state for dashboard."""
        current_date = datetime.now()
        
        # Calculate metrics
        total_drugs = len(self.drugs)
        active_suppliers = len(self.suppliers)
        
        # Fraud analysis
        fraud_alerts = sum(1 for tx in self.transactions if tx.get('is_fraudulent', False))
        new_fraud_alerts = sum(1 for tx in self.transactions 
                              if tx.get('is_fraudulent', False) and 
                              tx['timestamp'].date() == current_date.date())
        
        # Expiration analysis
        expiring_drugs = 0
        expiring_drugs_detail = []
        
        for key, item in self.inventory.items():
            days_to_expire = (item['expiry_date'] - current_date).days
            if days_to_expire <= 30:
                expiring_drugs += 1
                drug_info = next(d for d in self.drugs if d['id'] == item['drug_id'])
                expiring_drugs_detail.append({
                    'drug_name': drug_info['name'],
                    'hospital_id': item['hospital_id'],
                    'current_stock': item['current_stock'],
                    'days_to_expire': days_to_expire,
                    'batch_number': item['batch_number']
                })
        
        # Inventory levels
        inventory_levels = []
        for key, item in self.inventory.items():
            drug_info = next(d for d in self.drugs if d['id'] == item['drug_id'])
            stock_ratio = item['current_stock'] / item['max_stock']
            
            if stock_ratio < 0.2:
                status = 'Critical'
            elif stock_ratio < 0.4:
                status = 'Low'
            elif stock_ratio > 0.8:
                status = 'Overstocked'
            else:
                status = 'Normal'
            
            inventory_levels.append({
                'drug_name': drug_info['name'][:20],  # Truncate for display
                'current_stock': item['current_stock'],
                'stock_status': status,
                'hospital_id': item['hospital_id']
            })
        
        # Generate efficiency metrics
        efficiency_metrics = []
        for i in range(30):
            date = current_date - timedelta(days=29-i)
            efficiency_score = random.uniform(0.7, 0.95) + random.uniform(-0.1, 0.1)
            efficiency_metrics.append({
                'date': date,
                'efficiency_score': max(0, min(1, efficiency_score))
            })
        
        # Cost analysis
        total_cost = sum(tx['total_amount'] for tx in self.transactions)
        cost_analysis = {
            'categories': ['Drug Procurement', 'Storage', 'Transportation', 'Waste Management', 'Compliance'],
            'costs': [
                total_cost * 0.6,  # Drug procurement
                total_cost * 0.15, # Storage
                total_cost * 0.1,  # Transportation
                total_cost * 0.1,  # Waste management
                total_cost * 0.05  # Compliance
            ]
        }
        
        # Waste metrics
        waste_metrics = {
            'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'waste_prevented': [random.randint(100, 500) for _ in range(6)]
        }
        
        return {
            'total_drugs': total_drugs,
            'new_drugs_today': random.randint(0, 5),
            'active_suppliers': active_suppliers,
            'supplier_performance': random.uniform(0.8, 0.95),
            'fraud_alerts': fraud_alerts,
            'new_fraud_alerts': new_fraud_alerts,
            'expiring_drugs': expiring_drugs,
            'expired_prevented': random.randint(0, 10),
            'inventory_levels': pd.DataFrame(inventory_levels),
            'expiring_drugs_detail': pd.DataFrame(expiring_drugs_detail),
            'transactions': self.transactions,
            'efficiency_metrics': pd.DataFrame(efficiency_metrics),
            'cost_analysis': cost_analysis,
            'waste_metrics': waste_metrics
        }
