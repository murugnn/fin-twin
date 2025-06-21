import json
import csv
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Union
import re
from collections import defaultdict

class CashExpenseManager:
    def __init__(self, data_file: str = "cash_expenses.json"):
        """
        Initialize the Cash Expense Manager
        
        Args:
            data_file: Path to store cash transaction data (JSON or CSV)
        """
        self.data_file = data_file
        self.transactions = []
        self.categories = {
            'food': ['food', 'restaurant', 'cafe', 'lunch', 'dinner', 'breakfast', 'snack', 'street food', 'meal', 'eat', 'drink', 'coffee', 'tea', 'juice', 'bakery', 'pizza', 'burger'],
            'transport': ['bus', 'taxi', 'auto', 'rickshaw', 'metro', 'transport', 'fuel', 'parking', 'petrol', 'diesel', 'uber', 'ola', 'train', 'flight', 'cab', 'bike', 'car'],
            'utilities': ['rent', 'electricity', 'water', 'gas', 'maintenance', 'repair', 'bill', 'internet', 'phone', 'mobile', 'wifi', 'broadband', 'cable', 'dtv'],
            'shopping': ['clothes', 'grocery', 'market', 'shop', 'purchase', 'buy', 'store', 'mall', 'supermarket', 'clothing', 'shoes', 'accessories', 'electronics', 'appliance'],
            'entertainment': ['movie', 'game', 'party', 'concert', 'fun', 'entertainment', 'cinema', 'theater', 'club', 'bar', 'music', 'sports', 'event', 'ticket', 'subscription'],
            'health': ['medicine', 'doctor', 'hospital', 'pharmacy', 'medical', 'health', 'clinic', 'checkup', 'treatment', 'dental', 'surgery', 'insurance', 'therapy'],
            'personal': ['haircut', 'salon', 'personal', 'grooming', 'spa', 'beauty', 'cosmetics', 'barber', 'massage', 'skincare', 'personal care'],
            'education': ['education', 'school', 'college', 'university', 'course', 'book', 'tuition', 'fee', 'exam', 'training', 'workshop', 'certification', 'study'],
            'investment': ['investment', 'stock', 'mutual fund', 'fd', 'deposit', 'insurance', 'policy', 'sip', 'savings', 'portfolio', 'equity', 'bond'],
            'miscellaneous': []
        }
        self.load_data()

    def load_data(self):
        """Load existing transaction data from file"""
        try:
            if self.data_file.endswith('.json'):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.transactions = data.get('transactions', [])
            elif self.data_file.endswith('.csv'):
                df = pd.read_csv(self.data_file)
                self.transactions = df.to_dict('records')
        except FileNotFoundError:
            print(f"No existing data file found. Starting fresh.")
            self.transactions = []
        except Exception as e:
            print(f"Error loading data: {e}")
            self.transactions = []

    def save_data(self):
        """Save transaction data to file"""
        try:
            if self.data_file.endswith('.json'):
                with open(self.data_file, 'w') as f:
                    json.dump({'transactions': self.transactions}, f, indent=2, default=str)
            elif self.data_file.endswith('.csv'):
                df = pd.DataFrame(self.transactions)
                df.to_csv(self.data_file, index=False)
            print(f"Data saved to {self.data_file}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def auto_categorize(self, description: str) -> str:
        """
        Automatically categorize transaction based on keywords in description
        
        Args:
            description: Transaction description
            
        Returns:
            Predicted category
        """
        desc_lower = description.lower()
        
        for category, keywords in self.categories.items():
            if category == 'miscellaneous':
                continue
            for keyword in keywords:
                if keyword in desc_lower:
                    return category
        
        return 'miscellaneous'

    def add_transaction(self, 
                       description: str, 
                       amount: float, 
                       category: Optional[str] = None,
                       transaction_date: Optional[Union[str, date]] = None) -> bool:
        """
        Add a new cash transaction
        
        Args:
            description: Description of the transaction
            amount: Amount spent (positive number)
            category: Category (optional, will auto-categorize if not provided)
            transaction_date: Date of transaction (optional, uses today if not provided)
            
        Returns:
            True if transaction added successfully
        """
        try:
           
            if transaction_date is None:
                transaction_date = date.today()
            elif isinstance(transaction_date, str):
                transaction_date = datetime.strptime(transaction_date, '%Y-%m-%d').date()
            
           
            if category is None:
                category = self.auto_categorize(description)
            
           
            if amount <= 0:
                print("Amount must be positive")
                return False
            
            transaction = {
                'date': transaction_date.isoformat(),
                'description': description,
                'amount': float(amount),
                'category': category.lower(),
                'type': 'cash'
            }
            
            self.transactions.append(transaction)
            self.save_data()
            print(f"Transaction added: {description} - ${amount:.2f} ({category})")
            return True
            
        except Exception as e:
            print(f"Error adding transaction: {e}")
            return False

    def load_bank_data(self, bank_csv_path: str, 
                      amount_col: str = 'amount',
                      description_col: str = 'description',
                      date_col: str = 'date') -> bool:
        """
        Load digital/bank transaction data from CSV for combined analysis
        
        Args:
            bank_csv_path: Path to bank statement CSV
            amount_col: Column name for transaction amounts
            description_col: Column name for descriptions
            date_col: Column name for dates
            
        Returns:
            True if loaded successfully
        """
        try:
            df = pd.read_csv(bank_csv_path)
            
           
            bank_transactions = []
            for _, row in df.iterrows():
                transaction = {
                    'date': pd.to_datetime(row[date_col]).date().isoformat(),
                    'description': str(row[description_col]),
                    'amount': abs(float(row[amount_col])), 
                    'category': self.auto_categorize(str(row[description_col])),
                    'type': 'digital'
                }
                bank_transactions.append(transaction)
            
          
            cash_transactions = [t for t in self.transactions if t.get('type', 'cash') == 'cash']
            
           
            self.transactions = cash_transactions + bank_transactions
            
            print(f"Loaded {len(bank_transactions)} bank transactions")
            return True
            
        except Exception as e:
            print(f"Error loading bank data: {e}")
            return False

    def get_summary(self, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> Dict:
        """
        Get comprehensive expense summary
        
        Args:
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            
        Returns:
            Dictionary with summary statistics
        """
        filtered_transactions = self.transactions
        
       
        if start_date or end_date:
            filtered_transactions = []
            for t in self.transactions:
                t_date = datetime.strptime(t['date'], '%Y-%m-%d').date()
                if start_date and t_date < datetime.strptime(start_date, '%Y-%m-%d').date():
                    continue
                if end_date and t_date > datetime.strptime(end_date, '%Y-%m-%d').date():
                    continue
                filtered_transactions.append(t)
        
     
        cash_total = sum(t['amount'] for t in filtered_transactions if t.get('type', 'cash') == 'cash')
        digital_total = sum(t['amount'] for t in filtered_transactions if t.get('type', 'cash') == 'digital')
        total_expenses = cash_total + digital_total
        
        cash_by_category = defaultdict(float)
        digital_by_category = defaultdict(float)
        
        for t in filtered_transactions:
            if t.get('type', 'cash') == 'cash':
                cash_by_category[t['category']] += t['amount']
            else:
                digital_by_category[t['category']] += t['amount']
        
        summary = {
            'total_cash': cash_total,
            'total_digital': digital_total,
            'total_expenses': total_expenses,
            'cash_percentage': (cash_total / total_expenses * 100) if total_expenses > 0 else 0,
            'digital_percentage': (digital_total / total_expenses * 100) if total_expenses > 0 else 0,
            'cash_by_category': dict(cash_by_category),
            'digital_by_category': dict(digital_by_category),
            'transaction_count': {
                'cash': len([t for t in filtered_transactions if t.get('type', 'cash') == 'cash']),
                'digital': len([t for t in filtered_transactions if t.get('type', 'cash') == 'digital'])
            }
        }
        
        return summary

    def print_summary(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Print formatted summary to console"""
        summary = self.get_summary(start_date, end_date)
        
        print("\n" + "="*50)
        print("EXPENSE SUMMARY")
        print("="*50)
        print(f"Total Cash Expenses: ${summary['total_cash']:.2f}")
        print(f"Total Digital Expenses: ${summary['total_digital']:.2f}")
        print(f"Total Expenses: ${summary['total_expenses']:.2f}")
        print(f"\nCash vs Digital Ratio: {summary['cash_percentage']:.1f}% Cash, {summary['digital_percentage']:.1f}% Digital")
        
        print(f"\nTransaction Counts: {summary['transaction_count']['cash']} Cash, {summary['transaction_count']['digital']} Digital")
        
        print("\nCASH EXPENSES BY CATEGORY:")
        print("-" * 30)
        for category, amount in sorted(summary['cash_by_category'].items(), key=lambda x: x[1], reverse=True):
            print(f"{category.capitalize()}: ${amount:.2f}")
        
        if summary['digital_by_category']:
            print("\nDIGITAL EXPENSES BY CATEGORY:")
            print("-" * 30)
            for category, amount in sorted(summary['digital_by_category'].items(), key=lambda x: x[1], reverse=True):
                print(f"{category.capitalize()}: ${amount:.2f}")

    def simulate_savings(self, reduction_scenarios: Dict[str, float]) -> Dict:
        """
        Simulate savings by reducing cash expenses
        
        Args:
            reduction_scenarios: Dict with category/percentage reduction
                                e.g., {'food': 0.3, 'transport': 0.5} for 30% food, 50% transport reduction
                                
        Returns:
            Dictionary with simulation results
        """
        current_summary = self.get_summary()
        current_cash = current_summary['total_cash']
        
        total_savings = 0
        scenario_details = {}
        
        for category, reduction_pct in reduction_scenarios.items():
            if category in current_summary['cash_by_category']:
                category_amount = current_summary['cash_by_category'][category]
                savings = category_amount * reduction_pct
                total_savings += savings
                scenario_details[category] = {
                    'current_amount': category_amount,
                    'reduction_percentage': reduction_pct * 100,
                    'savings': savings,
                    'new_amount': category_amount - savings
                }
        
        new_cash_total = current_cash - total_savings
        new_total_expenses = current_summary['total_digital'] + new_cash_total
        
        simulation = {
            'current_cash_total': current_cash,
            'projected_cash_total': new_cash_total,
            'total_savings': total_savings,
            'savings_percentage': (total_savings / current_cash * 100) if current_cash > 0 else 0,
            'new_cash_percentage': (new_cash_total / new_total_expenses * 100) if new_total_expenses > 0 else 0,
            'scenario_details': scenario_details
        }
        
        return simulation

    def print_simulation(self, reduction_scenarios: Dict[str, float]):
        """Print savings simulation results"""
        sim = self.simulate_savings(reduction_scenarios)
        
        print("\n" + "="*50)
        print("SAVINGS SIMULATION")
        print("="*50)
        print(f"Current Cash Expenses: ${sim['current_cash_total']:.2f}")
        print(f"Projected Cash Expenses: ${sim['projected_cash_total']:.2f}")
        print(f"Total Potential Savings: ${sim['total_savings']:.2f} ({sim['savings_percentage']:.1f}%)")
        print(f"New Cash Percentage: {sim['new_cash_percentage']:.1f}%")
        
        print("\nSCENARIO BREAKDOWN:")
        print("-" * 30)
        for category, details in sim['scenario_details'].items():
            print(f"{category.capitalize()}:")
            print(f"  Current: ${details['current_amount']:.2f}")
            print(f"  Reduction: {details['reduction_percentage']:.0f}%")
            print(f"  Savings: ${details['savings']:.2f}")
            print(f"  New Amount: ${details['new_amount']:.2f}")
            print()

    def list_transactions(self, transaction_type: str = 'all', limit: int = 10):
        """
        List recent transactions
        
        Args:
            transaction_type: 'cash', 'digital', or 'all'
            limit: Number of transactions to show
        """
        transactions = self.transactions
        
        if transaction_type != 'all':
            transactions = [t for t in transactions if t.get('type', 'cash') == transaction_type]
       
        transactions = sorted(transactions, key=lambda x: x['date'], reverse=True)
        transactions = transactions[:limit]
        
        print(f"\nRECENT {transaction_type.upper()} TRANSACTIONS:")
        print("-" * 60)
        print(f"{'Date':<12} {'Description':<25} {'Amount':<10} {'Category':<15}")
        print("-" * 60)
        
        for t in transactions:
            print(f"{t['date']:<12} {t['description'][:24]:<25} ${t['amount']:<9.2f} {t['category']:<15}")

def cli_add_transaction(manager: CashExpenseManager):
    """CLI function to add a transaction"""
    print("\n--- Add New Cash Transaction ---")
    description = input("Description: ")
    
    try:
        amount = float(input("Amount: $"))
    except ValueError:
        print("Invalid amount entered")
        return
    
    category = input("Category (optional, press Enter to auto-categorize): ").strip()
    if not category:
        category = None
    
    date_input = input("Date (YYYY-MM-DD, press Enter for today): ").strip()
    if not date_input:
        date_input = None
    
    manager.add_transaction(description, amount, category, date_input)

def cli_main():
    """Main CLI interface"""
    manager = CashExpenseManager()
    
    while True:
        print("\n" + "="*40)
        print("CASH EXPENSE MANAGER")
        print("="*40)
        print("1. Add Transaction")
        print("2. View Summary")
        print("3. List Recent Transactions")
        print("4. Load Bank Data")
        print("5. Simulate Savings")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            cli_add_transaction(manager)
        
        elif choice == '2':
            start_date = input("Start date (YYYY-MM-DD, optional): ").strip() or None
            end_date = input("End date (YYYY-MM-DD, optional): ").strip() or None
            manager.print_summary(start_date, end_date)
        
        elif choice == '3':
            trans_type = input("Type (cash/digital/all): ").strip().lower() or 'all'
            try:
                limit = int(input("Number to show (default 10): ") or 10)
            except ValueError:
                limit = 10
            manager.list_transactions(trans_type, limit)
        
        elif choice == '4':
            csv_path = input("Path to bank CSV file: ").strip()
            if csv_path:
                manager.load_bank_data(csv_path)
        
        elif choice == '5':
            print("Enter reduction scenarios (category: percentage)")
            print("Example: food 0.3 (30% reduction in food expenses)")
            scenarios = {}
            while True:
                scenario = input("Category and percentage (or 'done' to finish): ").strip()
                if scenario.lower() == 'done':
                    break
                try:
                    parts = scenario.split()
                    if len(parts) == 2:
                        category, pct = parts[0], float(parts[1])
                        scenarios[category] = pct
                    else:
                        print("Invalid format. Use: category percentage")
                except ValueError:
                    print("Invalid percentage")
            
            if scenarios:
                manager.print_simulation(scenarios)
        
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    cli_main()