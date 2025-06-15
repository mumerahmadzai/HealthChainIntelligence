import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
import random
from dataclasses import dataclass, asdict
from enum import Enum

class TransactionType(Enum):
    DRUG_PROCUREMENT = "Drug Procurement"
    INVENTORY_TRANSFER = "Inventory Transfer"
    QUALITY_VERIFICATION = "Quality Verification"
    EXPIRATION_UPDATE = "Expiration Update"
    FRAUD_ALERT = "Fraud Alert"
    COMPLIANCE_CHECK = "Compliance Check"
    SUPPLIER_VERIFICATION = "Supplier Verification"
    PAYMENT_PROCESSING = "Payment Processing"

class TransactionStatus(Enum):
    PENDING = "Pending"
    CONFIRMED = "Confirmed"
    FAILED = "Failed"
    REJECTED = "Rejected"

class SmartContractEventType(Enum):
    DRUG_EXPIRED = "Drug Expired"
    STOCK_LOW = "Stock Low"
    FRAUD_DETECTED = "Fraud Detected"
    QUALITY_FAILED = "Quality Failed"
    RESTOCK_TRIGGERED = "Restock Triggered"
    EMERGENCY_ORDER = "Emergency Order"
    SUPPLIER_BLACKLISTED = "Supplier Blacklisted"
    COMPLIANCE_VIOLATION = "Compliance Violation"

@dataclass
class Block:
    index: int
    timestamp: datetime
    transactions: List[Dict[str, Any]]
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate the hash of the current block."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp.isoformat(),
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4):
        """Mine the block with proof of work."""
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()

@dataclass
class Transaction:
    id: str
    timestamp: datetime
    transaction_type: TransactionType
    from_address: str
    to_address: str
    data: Dict[str, Any]
    hash: str
    status: TransactionStatus
    gas_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.transaction_type.value,
            'from_address': self.from_address,
            'to_address': self.to_address,
            'data': self.data,
            'hash': self.hash,
            'status': self.status.value,
            'gas_used': self.gas_used
        }

@dataclass
class SmartContractEvent:
    event_id: str
    timestamp: datetime
    event_type: SmartContractEventType
    contract_address: str
    data: Dict[str, Any]
    transaction_hash: str
    block_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'contract_address': self.contract_address,
            'data': self.data,
            'transaction_hash': self.transaction_hash,
            'block_number': self.block_number
        }

class BlockchainSimulator:
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.transaction_pool: List[Transaction] = []
        self.smart_contract_events: List[SmartContractEvent] = []
        self.mining_difficulty = 4
        self.block_size_limit = 10  # Maximum transactions per block
        self.network_nodes = ['Node_001', 'Node_002', 'Node_003', 'Node_004', 'Node_005']
        self.active_nodes = len(self.network_nodes)
        self.gas_price = 20  # Gwei equivalent
        
        # Initialize blockchain with genesis block
        self._create_genesis_block()
        
        # Generate some initial transactions and blocks
        self._initialize_blockchain_data()
    
    def _create_genesis_block(self):
        """Create the first block in the blockchain."""
        genesis_block = Block(
            index=0,
            timestamp=datetime.now() - timedelta(days=30),
            transactions=[{
                'id': 'GENESIS',
                'type': 'Genesis Block',
                'data': 'Healthcare Supply Chain Blockchain Initialized',
                'hash': 'genesis_hash'
            }],
            previous_hash="0"
        )
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
    
    def _initialize_blockchain_data(self):
        """Initialize blockchain with historical data."""
        # Generate historical transactions
        for i in range(50):  # 50 historical transactions
            transaction = self._generate_sample_transaction()
            self.transaction_pool.append(transaction)
            
            # Mine blocks periodically
            if len(self.transaction_pool) >= self.block_size_limit:
                self._mine_pending_transactions()
        
        # Mine any remaining transactions
        if self.transaction_pool:
            self._mine_pending_transactions()
        
        # Generate smart contract events
        self._generate_smart_contract_events()
    
    def _generate_sample_transaction(self) -> Transaction:
        """Generate a sample transaction for simulation."""
        transaction_types = list(TransactionType)
        transaction_type = random.choice(transaction_types)
        
        # Generate addresses
        suppliers = [f'SUPPLIER_{i:03d}' for i in range(10)]
        hospitals = [f'HOSPITAL_{i:03d}' for i in range(25)]
        
        from_address = random.choice(suppliers + hospitals)
        to_address = random.choice(suppliers + hospitals)
        
        # Ensure from and to are different
        while to_address == from_address:
            to_address = random.choice(suppliers + hospitals)
        
        # Generate transaction data based on type
        data = self._generate_transaction_data(transaction_type)
        
        transaction_id = str(uuid.uuid4())
        timestamp = datetime.now() - timedelta(
            days=random.randint(1, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Create transaction hash
        transaction_string = json.dumps({
            'id': transaction_id,
            'timestamp': timestamp.isoformat(),
            'type': transaction_type.value,
            'from': from_address,
            'to': to_address,
            'data': data
        }, sort_keys=True)
        
        transaction_hash = hashlib.sha256(transaction_string.encode()).hexdigest()
        
        return Transaction(
            id=transaction_id,
            timestamp=timestamp,
            transaction_type=transaction_type,
            from_address=from_address,
            to_address=to_address,
            data=data,
            hash=transaction_hash,
            status=random.choice(list(TransactionStatus)),
            gas_used=random.randint(21000, 100000)
        )
    
    def _generate_transaction_data(self, transaction_type: TransactionType) -> Dict[str, Any]:
        """Generate transaction data based on transaction type."""
        drug_names = ['Amoxicillin', 'Ibuprofen', 'Insulin', 'Metformin', 'Lisinopril']
        
        if transaction_type == TransactionType.DRUG_PROCUREMENT:
            return {
                'drug_name': random.choice(drug_names),
                'quantity': random.randint(100, 1000),
                'unit_price': round(random.uniform(10.0, 500.0), 2),
                'batch_number': f'BATCH_{random.randint(100000, 999999)}',
                'expiry_date': (datetime.now() + timedelta(days=random.randint(180, 1095))).isoformat(),
                'supplier_certification': f'CERT_{random.randint(1000, 9999)}'
            }
        
        elif transaction_type == TransactionType.INVENTORY_TRANSFER:
            return {
                'drug_name': random.choice(drug_names),
                'quantity': random.randint(10, 500),
                'transfer_reason': random.choice(['Restock', 'Emergency', 'Redistribution']),
                'batch_number': f'BATCH_{random.randint(100000, 999999)}',
                'temperature_log': f'{random.uniform(2.0, 8.0):.1f}°C'
            }
        
        elif transaction_type == TransactionType.QUALITY_VERIFICATION:
            return {
                'drug_name': random.choice(drug_names),
                'batch_number': f'BATCH_{random.randint(100000, 999999)}',
                'test_results': random.choice(['Pass', 'Fail', 'Conditional Pass']),
                'quality_score': round(random.uniform(7.0, 10.0), 1),
                'inspector_id': f'QC_{random.randint(1, 20)}',
                'test_parameters': ['Visual Inspection', 'Potency Test', 'Contamination Check']
            }
        
        elif transaction_type == TransactionType.FRAUD_ALERT:
            return {
                'alert_type': random.choice(['Price Manipulation', 'Quantity Anomaly', 'Duplicate Shipment']),
                'risk_score': round(random.uniform(0.7, 1.0), 2),
                'suspicious_activity': 'Anomalous transaction pattern detected',
                'investigation_status': random.choice(['Open', 'Under Review', 'Resolved']),
                'affected_transactions': [f'TXN_{random.randint(100000, 999999)}' for _ in range(random.randint(1, 3))]
            }
        
        elif transaction_type == TransactionType.COMPLIANCE_CHECK:
            return {
                'regulation_type': random.choice(['FDA', 'DEA', 'State Pharmacy Board']),
                'compliance_status': random.choice(['Compliant', 'Non-Compliant', 'Under Review']),
                'audit_date': datetime.now().isoformat(),
                'findings': random.choice(['No issues found', 'Minor violations', 'Major violations']),
                'corrective_actions': ['Update documentation', 'Staff training', 'Process improvement']
            }
        
        else:
            return {
                'description': f'{transaction_type.value} transaction',
                'timestamp': datetime.now().isoformat(),
                'status': 'Processed'
            }
    
    def _mine_pending_transactions(self):
        """Mine pending transactions into a new block."""
        if not self.transaction_pool:
            return
        
        # Get transactions to include in block
        transactions_to_mine = self.transaction_pool[:self.block_size_limit]
        self.transaction_pool = self.transaction_pool[self.block_size_limit:]
        
        # Convert transactions to dictionary format for block
        transaction_dicts = [tx.to_dict() for tx in transactions_to_mine]
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now() - timedelta(
                minutes=random.randint(1, 30)  # Simulate block time
            ),
            transactions=transaction_dicts,
            previous_hash=self.chain[-1].hash
        )
        
        # Mine the block
        new_block.mine_block(self.mining_difficulty)
        
        # Add block to chain
        self.chain.append(new_block)
        
        # Generate smart contract events for this block
        self._generate_block_smart_contract_events(new_block, transactions_to_mine)
    
    def _generate_smart_contract_events(self):
        """Generate smart contract events based on blockchain state."""
        for _ in range(20):  # Generate 20 sample events
            event = self._generate_sample_smart_contract_event()
            self.smart_contract_events.append(event)
    
    def _generate_block_smart_contract_events(self, block: Block, transactions: List[Transaction]):
        """Generate smart contract events for a specific block."""
        for transaction in transactions:
            # Randomly generate events based on transaction type
            if random.random() < 0.3:  # 30% chance of generating an event
                event = self._generate_smart_contract_event_from_transaction(block, transaction)
                if event:
                    self.smart_contract_events.append(event)
    
    def _generate_sample_smart_contract_event(self) -> SmartContractEvent:
        """Generate a sample smart contract event."""
        event_types = list(SmartContractEventType)
        event_type = random.choice(event_types)
        
        contract_addresses = [
            '0x742d35Cc6634C0532925a3b8D5c3A8b07d5a1e8d',
            '0x1234567890123456789012345678901234567890',
            '0xabcdefabcdefabcdefabcdefabcdefabcdefabcd',
            '0x9876543210987654321098765432109876543210'
        ]
        
        event_data = self._generate_smart_contract_event_data(event_type)
        
        return SmartContractEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now() - timedelta(
                days=random.randint(1, 30),
                hours=random.randint(0, 23)
            ),
            event_type=event_type,
            contract_address=random.choice(contract_addresses),
            data=event_data,
            transaction_hash=hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest(),
            block_number=random.randint(1, len(self.chain))
        )
    
    def _generate_smart_contract_event_from_transaction(self, block: Block, transaction: Transaction) -> Optional[SmartContractEvent]:
        """Generate smart contract event from a specific transaction."""
        event_type = None
        
        # Determine event type based on transaction
        if transaction.transaction_type == TransactionType.FRAUD_ALERT:
            event_type = SmartContractEventType.FRAUD_DETECTED
        elif transaction.transaction_type == TransactionType.QUALITY_VERIFICATION:
            if transaction.data.get('test_results') == 'Fail':
                event_type = SmartContractEventType.QUALITY_FAILED
        elif transaction.transaction_type == TransactionType.DRUG_PROCUREMENT:
            if random.random() < 0.1:  # 10% chance of triggering restock
                event_type = SmartContractEventType.RESTOCK_TRIGGERED
        
        if not event_type:
            return None
        
        event_data = self._generate_smart_contract_event_data(event_type)
        event_data.update({
            'triggering_transaction': transaction.id,
            'related_data': transaction.data
        })
        
        return SmartContractEvent(
            event_id=str(uuid.uuid4()),
            timestamp=block.timestamp,
            event_type=event_type,
            contract_address='0x742d35Cc6634C0532925a3b8D5c3A8b07d5a1e8d',
            data=event_data,
            transaction_hash=transaction.hash,
            block_number=block.index
        )
    
    def _generate_smart_contract_event_data(self, event_type: SmartContractEventType) -> Dict[str, Any]:
        """Generate event data based on event type."""
        drug_names = ['Amoxicillin', 'Ibuprofen', 'Insulin', 'Metformin', 'Lisinopril']
        
        if event_type == SmartContractEventType.DRUG_EXPIRED:
            return {
                'drug_name': random.choice(drug_names),
                'batch_number': f'BATCH_{random.randint(100000, 999999)}',
                'expiry_date': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                'quantity_affected': random.randint(10, 500),
                'action_required': 'Remove from inventory'
            }
        
        elif event_type == SmartContractEventType.STOCK_LOW:
            return {
                'drug_name': random.choice(drug_names),
                'current_stock': random.randint(1, 20),
                'minimum_threshold': random.randint(25, 50),
                'hospital_id': f'HOSPITAL_{random.randint(1, 25):03d}',
                'reorder_recommended': True
            }
        
        elif event_type == SmartContractEventType.FRAUD_DETECTED:
            return {
                'fraud_type': random.choice(['Price Manipulation', 'Quantity Anomaly', 'Duplicate Shipment']),
                'risk_score': round(random.uniform(0.7, 1.0), 2),
                'supplier_id': f'SUPPLIER_{random.randint(1, 10):03d}',
                'investigation_initiated': True,
                'automated_response': 'Transaction flagged for review'
            }
        
        elif event_type == SmartContractEventType.QUALITY_FAILED:
            return {
                'drug_name': random.choice(drug_names),
                'batch_number': f'BATCH_{random.randint(100000, 999999)}',
                'test_failed': random.choice(['Potency', 'Contamination', 'Visual Inspection']),
                'quality_score': round(random.uniform(0.0, 6.9), 1),
                'action_taken': 'Batch quarantined'
            }
        
        elif event_type == SmartContractEventType.RESTOCK_TRIGGERED:
            return {
                'drug_name': random.choice(drug_names),
                'hospital_id': f'HOSPITAL_{random.randint(1, 25):03d}',
                'quantity_ordered': random.randint(100, 1000),
                'supplier_id': f'SUPPLIER_{random.randint(1, 10):03d}',
                'priority_level': random.choice(['Standard', 'High', 'Emergency'])
            }
        
        elif event_type == SmartContractEventType.EMERGENCY_ORDER:
            return {
                'drug_name': random.choice(drug_names),
                'hospital_id': f'HOSPITAL_{random.randint(1, 25):03d}',
                'urgency_level': 'Critical',
                'quantity_requested': random.randint(50, 500),
                'expected_delivery': (datetime.now() + timedelta(hours=random.randint(2, 24))).isoformat()
            }
        
        else:
            return {
                'event_description': f'{event_type.value} event occurred',
                'timestamp': datetime.now().isoformat(),
                'automated_processing': True
            }
    
    def add_transaction(self, transaction_type: TransactionType, from_address: str, 
                       to_address: str, data: Dict[str, Any]) -> str:
        """Add a new transaction to the pending pool."""
        transaction_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create transaction hash
        transaction_string = json.dumps({
            'id': transaction_id,
            'timestamp': timestamp.isoformat(),
            'type': transaction_type.value,
            'from': from_address,
            'to': to_address,
            'data': data
        }, sort_keys=True)
        
        transaction_hash = hashlib.sha256(transaction_string.encode()).hexdigest()
        
        transaction = Transaction(
            id=transaction_id,
            timestamp=timestamp,
            transaction_type=transaction_type,
            from_address=from_address,
            to_address=to_address,
            data=data,
            hash=transaction_hash,
            status=TransactionStatus.PENDING,
            gas_used=random.randint(21000, 100000)
        )
        
        self.pending_transactions.append(transaction)
        return transaction_id
    
    def get_status(self) -> Dict[str, Any]:
        """Get current blockchain status."""
        total_transactions = sum(len(block.transactions) for block in self.chain)
        verified_transactions = sum(
            1 for block in self.chain 
            for tx in block.transactions 
            if tx.get('status') == 'Confirmed'
        )
        
        # Calculate integrity score based on hash verification
        integrity_score = 1.0  # Assume perfect integrity for simulation
        if len(self.chain) > 1:
            valid_blocks = 0
            for i in range(1, len(self.chain)):
                if self.chain[i].previous_hash == self.chain[i-1].hash:
                    valid_blocks += 1
            integrity_score = valid_blocks / (len(self.chain) - 1) if len(self.chain) > 1 else 1.0
        
        return {
            'total_blocks': len(self.chain),
            'verified_transactions': verified_transactions,
            'pending_transactions': len(self.pending_transactions),
            'integrity_score': integrity_score,
            'active_nodes': self.active_nodes,
            'mining_difficulty': self.mining_difficulty,
            'last_block_time': self.chain[-1].timestamp if self.chain else None,
            'network_hash_rate': f'{random.randint(100, 1000)} TH/s',  # Simulated
            'blockchain_size': f'{len(self.chain) * 2.1:.1f} MB'  # Simulated
        }
    
    def get_recent_transactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent transactions from the blockchain."""
        all_transactions = []
        
        # Get transactions from recent blocks
        for block in reversed(self.chain):
            for tx in block.transactions:
                if tx.get('id') != 'GENESIS':  # Skip genesis transaction
                    all_transactions.append(tx)
                if len(all_transactions) >= limit:
                    break
            if len(all_transactions) >= limit:
                break
        
        # Add pending transactions
        for tx in self.pending_transactions:
            all_transactions.append(tx.to_dict())
            if len(all_transactions) >= limit:
                break
        
        return all_transactions[:limit]
    
    def get_contract_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get smart contract events."""
        # Sort events by timestamp (most recent first)
        sorted_events = sorted(
            self.smart_contract_events,
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        return [event.to_dict() for event in sorted_events[:limit]]
    
    def get_block_by_hash(self, block_hash: str) -> Optional[Dict[str, Any]]:
        """Get block by hash."""
        for block in self.chain:
            if block.hash == block_hash:
                return {
                    'index': block.index,
                    'timestamp': block.timestamp.isoformat(),
                    'transactions': block.transactions,
                    'previous_hash': block.previous_hash,
                    'hash': block.hash,
                    'nonce': block.nonce
                }
        return None
    
    def get_transaction_by_hash(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get transaction by hash."""
        # Search in blocks
        for block in self.chain:
            for tx in block.transactions:
                if tx.get('hash') == tx_hash:
                    return tx
        
        # Search in pending transactions
        for tx in self.pending_transactions:
            if tx.hash == tx_hash:
                return tx.to_dict()
        
        return None
    
    def verify_chain_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the blockchain."""
        integrity_issues = []
        
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Verify previous hash
            if current_block.previous_hash != previous_block.hash:
                integrity_issues.append({
                    'block_index': i,
                    'issue': 'Previous hash mismatch',
                    'expected': previous_block.hash,
                    'actual': current_block.previous_hash
                })
            
            # Verify block hash
            calculated_hash = current_block.calculate_hash()
            if current_block.hash != calculated_hash:
                integrity_issues.append({
                    'block_index': i,
                    'issue': 'Block hash invalid',
                    'expected': calculated_hash,
                    'actual': current_block.hash
                })
        
        return {
            'is_valid': len(integrity_issues) == 0,
            'total_blocks_checked': len(self.chain),
            'integrity_issues': integrity_issues,
            'integrity_score': 1.0 - (len(integrity_issues) / max(1, len(self.chain)))
        }
    
    def simulate_mining(self):
        """Simulate mining of pending transactions."""
        if len(self.pending_transactions) >= self.block_size_limit:
            # Move pending transactions to transaction pool
            self.transaction_pool.extend(self.pending_transactions[:self.block_size_limit])
            self.pending_transactions = self.pending_transactions[self.block_size_limit:]
            
            # Mine the block
            self._mine_pending_transactions()
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        total_transactions = sum(len(block.transactions) for block in self.chain)
        
        return {
            'total_blocks': len(self.chain),
            'total_transactions': total_transactions,
            'pending_transactions': len(self.pending_transactions),
            'active_nodes': self.active_nodes,
            'average_block_time': '10 minutes',  # Simulated
            'transactions_per_second': round(total_transactions / max(1, (datetime.now() - self.chain[0].timestamp).total_seconds()), 2),
            'blockchain_size_mb': round(len(self.chain) * 2.1, 2),
            'mining_difficulty': self.mining_difficulty,
            'network_uptime': '99.9%'  # Simulated
        }
