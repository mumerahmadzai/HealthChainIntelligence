import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum

class RuleStatus(Enum):
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    TRIGGERED = "Triggered"
    DISABLED = "Disabled"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Rule:
    id: str
    name: str
    description: str
    condition: Callable
    action: Callable
    priority: Priority
    status: RuleStatus
    trigger_count: int = 0
    last_triggered: datetime = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class RulesEngine:
    def __init__(self):
        self.rules = {}
        self.execution_history = []
        self.performance_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'rule_triggers': {},
            'average_execution_time': 0.0
        }
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default supply chain rules."""
        
        # Expiration Warning Rule
        self.add_rule(
            rule_id="EXP_001",
            name="Drug Expiration Warning",
            description="Alert when drugs are approaching expiration date",
            condition=self._check_expiration_condition,
            action=self._expiration_alert_action,
            priority=Priority.HIGH
        )
        
        # Low Stock Alert Rule
        self.add_rule(
            rule_id="STK_001",
            name="Low Stock Alert",
            description="Alert when inventory falls below minimum threshold",
            condition=self._check_low_stock_condition,
            action=self._low_stock_alert_action,
            priority=Priority.MEDIUM
        )
        
        # High Fraud Score Rule
        self.add_rule(
            rule_id="FRD_001",
            name="High Fraud Score Alert",
            description="Alert when transaction fraud score exceeds threshold",
            condition=self._check_fraud_score_condition,
            action=self._fraud_alert_action,
            priority=Priority.CRITICAL
        )
        
        # Overstocking Rule
        self.add_rule(
            rule_id="STK_002",
            name="Overstocking Alert",
            description="Alert when inventory exceeds maximum threshold",
            condition=self._check_overstock_condition,
            action=self._overstock_alert_action,
            priority=Priority.LOW
        )
        
        # Supplier Performance Rule
        self.add_rule(
            rule_id="SUP_001",
            name="Supplier Performance Degradation",
            description="Alert when supplier performance drops below acceptable level",
            condition=self._check_supplier_performance_condition,
            action=self._supplier_performance_action,
            priority=Priority.MEDIUM
        )
        
        # Emergency Restock Rule
        self.add_rule(
            rule_id="STK_003",
            name="Emergency Restock Required",
            description="Trigger emergency restock for critical drugs",
            condition=self._check_emergency_restock_condition,
            action=self._emergency_restock_action,
            priority=Priority.CRITICAL
        )
        
        # Quality Control Rule
        self.add_rule(
            rule_id="QC_001",
            name="Quality Control Failure",
            description="Alert when quality control checks fail",
            condition=self._check_quality_control_condition,
            action=self._quality_control_action,
            priority=Priority.HIGH
        )
        
        # Price Anomaly Rule
        self.add_rule(
            rule_id="PRC_001",
            name="Price Anomaly Detection",
            description="Alert when drug prices deviate significantly from normal",
            condition=self._check_price_anomaly_condition,
            action=self._price_anomaly_action,
            priority=Priority.MEDIUM
        )
    
    def add_rule(self, rule_id: str, name: str, description: str, condition: Callable, 
                 action: Callable, priority: Priority):
        """Add a new rule to the engine."""
        rule = Rule(
            id=rule_id,
            name=name,
            description=description,
            condition=condition,
            action=action,
            priority=priority,
            status=RuleStatus.ACTIVE
        )
        self.rules[rule_id] = rule
        self.performance_metrics['rule_triggers'][rule_id] = 0
    
    def execute_rules(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute all active rules against the provided data."""
        execution_results = []
        start_time = datetime.now()
        
        for rule_id, rule in self.rules.items():
            if rule.status != RuleStatus.ACTIVE:
                continue
            
            try:
                rule_start_time = datetime.now()
                
                # Check condition
                condition_result = rule.condition(data)
                
                if condition_result:
                    # Execute action
                    action_result = rule.action(data, condition_result)
                    
                    # Update rule statistics
                    rule.trigger_count += 1
                    rule.last_triggered = datetime.now()
                    rule.status = RuleStatus.TRIGGERED
                    self.performance_metrics['rule_triggers'][rule_id] += 1
                    
                    # Record execution
                    execution_time = (datetime.now() - rule_start_time).total_seconds()
                    
                    execution_result = {
                        'rule_id': rule_id,
                        'rule_name': rule.name,
                        'priority': rule.priority.name,
                        'triggered': True,
                        'condition_result': condition_result,
                        'action_result': action_result,
                        'execution_time': execution_time,
                        'timestamp': datetime.now()
                    }
                    
                    execution_results.append(execution_result)
                    self.execution_history.append(execution_result)
                    self.performance_metrics['successful_executions'] += 1
                else:
                    # Reset rule status if not triggered
                    if rule.status == RuleStatus.TRIGGERED:
                        rule.status = RuleStatus.ACTIVE
                
                self.performance_metrics['total_executions'] += 1
                
            except Exception as e:
                # Handle rule execution errors
                error_result = {
                    'rule_id': rule_id,
                    'rule_name': rule.name,
                    'priority': rule.priority.name,
                    'triggered': False,
                    'error': str(e),
                    'timestamp': datetime.now()
                }
                
                execution_results.append(error_result)
                self.performance_metrics['failed_executions'] += 1
                self.performance_metrics['total_executions'] += 1
        
        # Update average execution time
        total_time = (datetime.now() - start_time).total_seconds()
        if self.performance_metrics['total_executions'] > 0:
            self.performance_metrics['average_execution_time'] = (
                (self.performance_metrics['average_execution_time'] * 
                 (self.performance_metrics['total_executions'] - len(execution_results)) + 
                 total_time) / self.performance_metrics['total_executions']
            )
        
        # Keep execution history limited
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
        
        return execution_results
    
    # Rule Condition Functions
    def _check_expiration_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for drugs approaching expiration."""
        expiring_drugs = data.get('expiring_drugs_detail', pd.DataFrame())
        
        if expiring_drugs.empty:
            return None
        
        critical_expiring = expiring_drugs[expiring_drugs['days_to_expire'] <= 7]
        warning_expiring = expiring_drugs[expiring_drugs['days_to_expire'] <= 30]
        
        if len(critical_expiring) > 0 or len(warning_expiring) > 5:
            return {
                'critical_count': len(critical_expiring),
                'warning_count': len(warning_expiring),
                'critical_drugs': critical_expiring.to_dict('records'),
                'warning_drugs': warning_expiring.to_dict('records')
            }
        
        return None
    
    def _check_low_stock_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for low stock levels."""
        inventory_levels = data.get('inventory_levels', pd.DataFrame())
        
        if inventory_levels.empty:
            return None
        
        low_stock = inventory_levels[inventory_levels['stock_status'].isin(['Critical', 'Low'])]
        
        if len(low_stock) > 0:
            return {
                'low_stock_count': len(low_stock),
                'critical_count': len(low_stock[low_stock['stock_status'] == 'Critical']),
                'low_stock_items': low_stock.to_dict('records')
            }
        
        return None
    
    def _check_fraud_score_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for high fraud scores in transactions."""
        fraud_alerts = data.get('fraud_alerts', 0)
        new_fraud_alerts = data.get('new_fraud_alerts', 0)
        
        if fraud_alerts > 10 or new_fraud_alerts > 3:
            return {
                'total_fraud_alerts': fraud_alerts,
                'new_fraud_alerts': new_fraud_alerts,
                'severity': 'High' if new_fraud_alerts > 5 else 'Medium'
            }
        
        return None
    
    def _check_overstock_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for overstocking."""
        inventory_levels = data.get('inventory_levels', pd.DataFrame())
        
        if inventory_levels.empty:
            return None
        
        overstocked = inventory_levels[inventory_levels['stock_status'] == 'Overstocked']
        
        if len(overstocked) > 5:
            return {
                'overstocked_count': len(overstocked),
                'overstocked_items': overstocked.to_dict('records')
            }
        
        return None
    
    def _check_supplier_performance_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check supplier performance."""
        supplier_performance = data.get('supplier_performance', 0.9)
        
        if supplier_performance < 0.8:
            return {
                'performance_score': supplier_performance,
                'threshold': 0.8,
                'deviation': 0.8 - supplier_performance
            }
        
        return None
    
    def _check_emergency_restock_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for emergency restock situations."""
        inventory_levels = data.get('inventory_levels', pd.DataFrame())
        
        if inventory_levels.empty:
            return None
        
        critical_stock = inventory_levels[inventory_levels['stock_status'] == 'Critical']
        
        # Check if critical drugs are involved
        emergency_items = []
        for _, item in critical_stock.iterrows():
            # Simulate critical drug check
            if np.random.random() < 0.3:  # 30% chance of being critical
                emergency_items.append(item.to_dict())
        
        if len(emergency_items) > 0:
            return {
                'emergency_count': len(emergency_items),
                'emergency_items': emergency_items
            }
        
        return None
    
    def _check_quality_control_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check quality control issues."""
        # Simulate quality control checks
        if np.random.random() < 0.05:  # 5% chance of quality issues
            return {
                'issue_type': np.random.choice(['Contamination', 'Packaging Defect', 'Temperature Breach']),
                'severity': np.random.choice(['Low', 'Medium', 'High']),
                'affected_batches': np.random.randint(1, 5)
            }
        
        return None
    
    def _check_price_anomaly_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for price anomalies."""
        transactions = data.get('transactions', [])
        
        if not transactions:
            return None
        
        # Simulate price anomaly detection
        if np.random.random() < 0.1:  # 10% chance of price anomaly
            return {
                'anomaly_type': np.random.choice(['Price Spike', 'Price Drop', 'Irregular Pricing']),
                'affected_drugs': np.random.randint(1, 3),
                'price_deviation': np.random.uniform(0.2, 0.8)
            }
        
        return None
    
    # Rule Action Functions
    def _expiration_alert_action(self, data: Dict[str, Any], condition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Action for expiration alerts."""
        return {
            'action_type': 'Expiration Alert',
            'message': f"Expiration alert: {condition_result['critical_count']} critical, {condition_result['warning_count']} warning",
            'recommendations': [
                'Review critical expiring drugs immediately',
                'Initiate emergency use protocols if applicable',
                'Contact suppliers for replacement stock'
            ],
            'priority': 'High' if condition_result['critical_count'] > 0 else 'Medium'
        }
    
    def _low_stock_alert_action(self, data: Dict[str, Any], condition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Action for low stock alerts."""
        return {
            'action_type': 'Low Stock Alert',
            'message': f"Low stock alert: {condition_result['low_stock_count']} items need restocking",
            'recommendations': [
                'Generate purchase orders for low stock items',
                'Prioritize critical medications',
                'Review demand forecasting models'
            ],
            'priority': 'High' if condition_result['critical_count'] > 0 else 'Medium'
        }
    
    def _fraud_alert_action(self, data: Dict[str, Any], condition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Action for fraud alerts."""
        return {
            'action_type': 'Fraud Alert',
            'message': f"Fraud alert: {condition_result['new_fraud_alerts']} new suspicious transactions",
            'recommendations': [
                'Review flagged transactions immediately',
                'Contact suppliers for verification',
                'Suspend suspicious accounts if necessary',
                'Escalate to security team'
            ],
            'priority': 'Critical'
        }
    
    def _overstock_alert_action(self, data: Dict[str, Any], condition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Action for overstock alerts."""
        return {
            'action_type': 'Overstock Alert',
            'message': f"Overstock alert: {condition_result['overstocked_count']} items overstocked",
            'recommendations': [
                'Review overstocked items for redistribution',
                'Adjust ordering quantities',
                'Consider promotional programs',
                'Check storage capacity'
            ],
            'priority': 'Low'
        }
    
    def _supplier_performance_action(self, data: Dict[str, Any], condition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Action for supplier performance issues."""
        return {
            'action_type': 'Supplier Performance Alert',
            'message': f"Supplier performance below threshold: {condition_result['performance_score']:.2%}",
            'recommendations': [
                'Review supplier contracts',
                'Initiate performance improvement discussions',
                'Consider alternative suppliers',
                'Implement additional monitoring'
            ],
            'priority': 'Medium'
        }
    
    def _emergency_restock_action(self, data: Dict[str, Any], condition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Action for emergency restock situations."""
        return {
            'action_type': 'Emergency Restock',
            'message': f"Emergency restock required for {condition_result['emergency_count']} critical items",
            'recommendations': [
                'Initiate emergency procurement procedures',
                'Contact priority suppliers',
                'Expedite shipping arrangements',
                'Notify clinical staff of potential shortages'
            ],
            'priority': 'Critical'
        }
    
    def _quality_control_action(self, data: Dict[str, Any], condition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Action for quality control issues."""
        return {
            'action_type': 'Quality Control Alert',
            'message': f"Quality issue detected: {condition_result['issue_type']}",
            'recommendations': [
                'Quarantine affected batches immediately',
                'Notify regulatory authorities if required',
                'Investigate root cause',
                'Review quality control procedures'
            ],
            'priority': 'High'
        }
    
    def _price_anomaly_action(self, data: Dict[str, Any], condition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Action for price anomalies."""
        return {
            'action_type': 'Price Anomaly Alert',
            'message': f"Price anomaly detected: {condition_result['anomaly_type']}",
            'recommendations': [
                'Review pricing agreements with suppliers',
                'Verify market conditions',
                'Check for data entry errors',
                'Consider renegotiating contracts'
            ],
            'priority': 'Medium'
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get rules engine performance metrics."""
        active_rules = []
        for rule_id, rule in self.rules.items():
            active_rules.append({
                'id': rule_id,
                'name': rule.name,
                'status': rule.status.value,
                'trigger_count': rule.trigger_count,
                'last_triggered': rule.last_triggered.strftime('%Y-%m-%d %H:%M:%S') if rule.last_triggered else 'Never'
            })
        
        # Calculate overall efficiency
        if self.performance_metrics['total_executions'] > 0:
            success_rate = self.performance_metrics['successful_executions'] / self.performance_metrics['total_executions']
            overall_efficiency = success_rate * 100
        else:
            overall_efficiency = 100
        
        return {
            'rule_triggers': self.performance_metrics['rule_triggers'],
            'overall_efficiency': overall_efficiency,
            'active_rules': active_rules,
            'total_executions': self.performance_metrics['total_executions'],
            'successful_executions': self.performance_metrics['successful_executions'],
            'failed_executions': self.performance_metrics['failed_executions'],
            'average_execution_time': self.performance_metrics['average_execution_time']
        }
    
    def get_rule_status(self, rule_id: str) -> Dict[str, Any]:
        """Get status of a specific rule."""
        if rule_id not in self.rules:
            return None
        
        rule = self.rules[rule_id]
        return {
            'id': rule.id,
            'name': rule.name,
            'description': rule.description,
            'status': rule.status.value,
            'priority': rule.priority.name,
            'trigger_count': rule.trigger_count,
            'last_triggered': rule.last_triggered,
            'created_at': rule.created_at
        }
    
    def enable_rule(self, rule_id: str):
        """Enable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id].status = RuleStatus.ACTIVE
    
    def disable_rule(self, rule_id: str):
        """Disable a rule."""
        if rule_id in self.rules:
            self.rules[rule_id].status = RuleStatus.DISABLED
