import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import re
import os
import sys
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import pdfplumber
    import tabula
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("PDF support not available. Install pdfplumber and tabula-py for PDF processing.")

class ExpenseAnalyzer:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.categories = ['Food', 'Transport', 'Subscriptions', 'Shopping', 'Utilities', 'Travel', 'Others']
        
        self.category_keywords = {
            'Food': ['zomato', 'swiggy', 'dominos', 'pizza', 'restaurant', 'food', 'cafe', 'coffee', 
                    'starbucks', 'mcdonalds', 'kfc', 'burger', 'bakery', 'dining', 'meal', 'lunch', 
                    'dinner', 'breakfast', 'grocery', 'supermarket', 'big bazaar', 'dmart'],
            
            'Transport': ['uber', 'ola', 'rapido', 'metro', 'bus', 'taxi', 'auto', 'fuel', 'petrol', 
                         'diesel', 'parking', 'toll', 'railway', 'train', 'flight', 'airline', 
                         'indigo', 'spicejet', 'vistara'],
            
            'Subscriptions': ['netflix', 'amazon prime', 'hotstar', 'spotify', 'youtube premium', 
                             'jio', 'airtel', 'vodafone', 'recharge', 'subscription', 'premium', 
                             'adobe', 'microsoft', 'google one', 'icloud'],
            
            'Shopping': ['amazon', 'flipkart', 'myntra', 'ajio', 'shopping', 'mall', 'store', 
                        'clothes', 'fashion', 'electronics', 'mobile', 'laptop', 'book', 
                        'cosmetics', 'pharmacy', 'medicine'],
            
            'Utilities': ['electricity', 'water', 'gas', 'internet', 'broadband', 'wifi', 'bill', 
                         'payment', 'utility', 'maintenance', 'society', 'rent', 'emi'],
            
            'Travel': ['hotel', 'booking', 'makemytrip', 'goibibo', 'oyo', 'travel', 'vacation', 
                      'trip', 'holiday', 'resort', 'tourism'],
        }
    
    def pdf_to_csv(self, pdf_path, output_csv_path=None):
        """Convert PDF bank statement to CSV format"""
        if not PDF_SUPPORT:
            print("PDF processing not available. Please install required libraries.")
            return False
        
        try:
            print(f"Processing PDF: {pdf_path}")
            
            if output_csv_path is None:
                output_csv_path = pdf_path.replace('.pdf', '_converted.csv')
            
            try:
                print("Attempting table extraction...")
                tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
                
                if tables:
                    
                    combined_df = pd.concat(tables, ignore_index=True)
                    
                    
                    combined_df.to_csv(output_csv_path, index=False)
                    print(f"PDF converted to CSV: {output_csv_path}")
                    return output_csv_path
                    
            except Exception as e:
                print(f"Table extraction failed: {str(e)}")
            
            
            try:
                print("Attempting text extraction...")
                extracted_data = []
                
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                           
                            lines = text.split('\n')
                            for line in lines:
                                
                                if self._is_transaction_line(line):
                                    parsed = self._parse_transaction_line(line)
                                    if parsed:
                                        extracted_data.append(parsed)
                
                if extracted_data:
                    df = pd.DataFrame(extracted_data)
                    df.to_csv(output_csv_path, index=False)
                    print(f"PDF converted to CSV: {output_csv_path}")
                    return output_csv_path
                else:
                    print("No transaction data found in PDF")
                    return False
                    
            except Exception as e:
                print(f"Text extraction failed: {str(e)}")
                return False
                
        except Exception as e:
            print(f"PDF processing error: {str(e)}")
            return False
    
    def _is_transaction_line(self, line):
        """Check if a line contains transaction data"""
        
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        amount_pattern = r'[\d,]+\.?\d*'
        
        has_date = bool(re.search(date_pattern, line))
        has_amount = bool(re.search(amount_pattern, line))
        
        return has_date and has_amount and len(line.strip()) > 20
    
    def _parse_transaction_line(self, line):
        """Parse a transaction line to extract date, description, amount"""
        try:

            parts = line.strip().split()
            
            
            date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
            date_match = re.search(date_pattern, line)
            
            
            amount_pattern = r'[\d,]+\.?\d*$'
            amount_match = re.search(amount_pattern, line)
            
            if date_match and amount_match:
                date_str = date_match.group()
                amount_str = amount_match.group().replace(',', '')
                
               
                desc_start = date_match.end()
                desc_end = amount_match.start()
                description = line[desc_start:desc_end].strip()
                
                return {
                    'Date': date_str,
                    'Description': description,
                    'Amount': float(amount_str)
                }
        except:
            pass
        
        return None
    
    def get_file_input(self):
        """Get file input from user"""
        print("\nEXPENSE ANALYZER - FILE INPUT")
        print("=" * 40)
        print("Supported formats: CSV, PDF")
        print("Enter the file path (or drag and drop the file):")
        
        while True:
            file_path = input("File path: ").strip().strip('"').strip("'")
            
            if not file_path:
                print("Please enter a valid file path")
                continue
            
            if not os.path.exists(file_path):
                print("File not found. Please check the path and try again.")
                continue
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                return file_path, 'csv'
            elif file_ext == '.pdf':
                if not PDF_SUPPORT:
                    print("PDF support not available. Please install pdfplumber and tabula-py.")
                    continue
                return file_path, 'pdf'
            else:
                print("Unsupported file format. Please use CSV or PDF files.")
                continue
    
    def load_csv(self, file_path):
        """Load and clean CSV file"""
        try:
            print(f"Loading CSV file: {file_path}")
            
            
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(file_path, encoding=encoding)
                    print(f"File loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.data is None:
                raise ValueError("Could not read the CSV file with any encoding")
            
            print(f"Raw data shape: {self.data.shape}")
            print(f"Columns found: {list(self.data.columns)}")
            
            
            self.data.columns = self.data.columns.str.strip().str.lower()
            
            
            column_mapping = {
                'transaction date': 'date',
                'trans date': 'date',
                'transaction_date': 'date',
                'txn date': 'date',
                'txn_date': 'date',
                'description': 'description',
                'narration': 'description',
                'particulars': 'description',
                'details': 'description',
                'transaction details': 'description',
                'amount': 'amount',
                'amount (inr)': 'amount',
                'debit': 'amount',
                'withdrawal': 'amount',
                'spent': 'amount',
                'debit amount': 'amount',
                'withdrawal amount': 'amount'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in self.data.columns:
                    self.data.rename(columns={old_col: new_col}, inplace=True)
            
            print(f"Mapped columns: {list(self.data.columns)}")
            
         
            required_columns = ['date', 'description', 'amount']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                print("Available columns:", list(self.data.columns))
                print("\nPlease ensure your CSV has columns for:")
                print("   - Date (transaction date)")
                print("   - Description (transaction details)")
                print("   - Amount (transaction amount)")
                return False
            
            
            self._clean_data()
            print(f"Successfully loaded {len(self.data)} transactions")
            return True
            
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            return False
    
    def _clean_data(self):
        """Clean and preprocess the data"""
        print("Cleaning data...")
        
        original_count = len(self.data)
        
       
        self.data = self.data.dropna(subset=['date', 'description', 'amount'])
        print(f"Removed {original_count - len(self.data)} rows with missing data")
        
        
        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        
       
        if self.data['date'].isna().any():
            print("Trying alternative date formats...")
            self.data['date'] = pd.to_datetime(self.data['date'], infer_datetime_format=True, errors='coerce')
        
     
        invalid_dates = self.data['date'].isna().sum()
        if invalid_dates > 0:
            print(f"Removed {invalid_dates} rows with invalid dates")
            self.data = self.data.dropna(subset=['date'])
        
      
        self.data['amount'] = pd.to_numeric(self.data['amount'], errors='coerce')
        invalid_amounts = self.data['amount'].isna().sum()
        if invalid_amounts > 0:
            print(f"Removed {invalid_amounts} rows with invalid amounts")
            self.data = self.data.dropna(subset=['amount'])
        
       
        negative_count = (self.data['amount'] < 0).sum()
        if negative_count > 0:
            print(f"Converting {negative_count} negative amounts to positive")
            self.data['amount'] = self.data['amount'].abs()
        
      
        small_amounts = (self.data['amount'] < 1).sum()
        if small_amounts > 0:
            print(f"Removed {small_amounts} transactions under ₹1")
            self.data = self.data[self.data['amount'] >= 1]
        
      
        self.data['description'] = self.data['description'].astype(str).str.lower().str.strip()
        
     
        self.data['month'] = self.data['date'].dt.to_period('M')
        self.data['week'] = self.data['date'].dt.to_period('W')
        self.data['day_of_week'] = self.data['date'].dt.day_name()
  
        self.data = self.data.sort_values('date').reset_index(drop=True)
        
        print(f"Data cleaned. Final dataset: {len(self.data)} transactions")
    
    def categorize_expenses(self):
        """Categorize expenses using rule-based approach"""
        print("Categorizing expenses...")
        
        def get_category(description):
            description = str(description).lower()
            
         
            for category, keywords in self.category_keywords.items():
                for keyword in keywords:
                    if keyword in description:
                        return category
            
            return 'Others'
        
        self.data['category'] = self.data['description'].apply(get_category)
        
     
        category_counts = self.data['category'].value_counts()
        print("Categorization Summary:")
        for category, count in category_counts.items():
            print(f"   {category}: {count} transactions")
        
        print("Expenses categorized successfully")
    
    def calculate_summaries(self):
        """Calculate various summaries and statistics"""
        print("Calculating summaries...")
        
        summaries = {}
   
        summaries['total_spending'] = self.data['amount'].sum()
        
      
        category_spending = self.data.groupby('category')['amount'].sum().sort_values(ascending=False)
        summaries['category_spending'] = category_spending.to_dict()
        
        monthly_spending = self.data.groupby('month')['amount'].sum()
        summaries['monthly_spending'] = {str(k): v for k, v in monthly_spending.to_dict().items()}
       
        weekly_spending = self.data.groupby('week')['amount'].sum()
        summaries['weekly_spending'] = {str(k): v for k, v in weekly_spending.to_dict().items()}
        
        daily_spending = self.data.groupby(self.data['date'].dt.date)['amount'].sum()
        summaries['daily_spending'] = {str(k): v for k, v in daily_spending.to_dict().items()}
        
        summaries['top_categories'] = list(category_spending.head(3).index)

        summaries['avg_daily_spending'] = daily_spending.mean()
        summaries['avg_transaction_amount'] = self.data['amount'].mean()
       
        summaries['total_transactions'] = len(self.data)
        summaries['transactions_per_category'] = self.data['category'].value_counts().to_dict()
        
        
        summaries['date_range'] = {
            'start': str(self.data['date'].min().date()),
            'end': str(self.data['date'].max().date()),
            'days': (self.data['date'].max() - self.data['date'].min()).days + 1
        }
        
        self.summaries = summaries
        print("Summaries calculated")
        return summaries
    
    def generate_insights(self):
        """Generate meaningful insights from the data"""
        print("Generating insights...")
        
        insights = []
        
        if not hasattr(self, 'summaries'):
            self.calculate_summaries()
        
        top_category = list(self.summaries['category_spending'].keys())[0]
        top_amount = list(self.summaries['category_spending'].values())[0]
        top_percentage = (top_amount / self.summaries['total_spending']) * 100
        
        insights.append({
            'type': 'top_category',
            'title': f'Your biggest expense: {top_category}',
            'message': f'You spent ₹{top_amount:,.0f} on {top_category} ({top_percentage:.1f}% of total spending)',
            'icon': '🍔' if top_category == 'Food' else '🚗' if top_category == 'Transport' else '📱' if top_category == 'Subscriptions' else '🛍️'
        })
        
   
        daily_avg = self.summaries['avg_daily_spending']
        recent_week = self.data[self.data['date'] >= self.data['date'].max() - timedelta(days=7)]
        
        if len(recent_week) > 0:
            recent_avg = recent_week['amount'].sum() / 7
            
            if recent_avg > daily_avg * 1.3:
                insights.append({
                    'type': 'high_spending',
                    'title': 'High spending alert! 📈',
                    'message': f'You spent ₹{recent_avg:.0f}/day this week vs usual ₹{daily_avg:.0f}/day',
                    'icon': '⚠️'
                })
            elif recent_avg < daily_avg * 0.7:
                insights.append({
                    'type': 'good_saving',
                    'title': 'Great job saving! 💰',
                    'message': f'You spent only ₹{recent_avg:.0f}/day this week vs usual ₹{daily_avg:.0f}/day',
                    'icon': '✅'
                })
        
        weekend_data = self.data[self.data['day_of_week'].isin(['Saturday', 'Sunday'])]
        weekday_data = self.data[~self.data['day_of_week'].isin(['Saturday', 'Sunday'])]
        
        if len(weekend_data) > 0 and len(weekday_data) > 0:
            weekend_spending = weekend_data['amount'].mean()
            weekday_spending = weekday_data['amount'].mean()
            
            if weekend_spending > weekday_spending * 1.5:
                insights.append({
                    'type': 'weekend_spender',
                    'title': 'Weekend splurger! 🎉',
                    'message': f'You spend ₹{weekend_spending:.0f} on weekends vs ₹{weekday_spending:.0f} on weekdays',
                    'icon': '🎭'
                })
        
        
        subscription_amount = self.summaries['category_spending'].get('Subscriptions', 0)
        if subscription_amount > 0:
            monthly_subscription = subscription_amount / self.summaries['date_range']['days'] * 30
            insights.append({
                'type': 'subscriptions',
                'title': 'Monthly subscriptions',
                'message': f'You spend ~₹{monthly_subscription:.0f}/month on subscriptions',
                'icon': '📺'
            })
    
        food_transactions = self.data[self.data['category'] == 'Food']
        food_delivery_keywords = ['zomato', 'swiggy', 'delivery']
        
        if len(food_transactions) > 0:
            delivery_transactions = food_transactions[
                food_transactions['description'].str.contains('|'.join(food_delivery_keywords), na=False)
            ]
            
            if len(delivery_transactions) > 5:
                avg_delivery_amount = delivery_transactions['amount'].mean()
                insights.append({
                    'type': 'food_delivery',
                    'title': 'Food delivery habits',
                    'message': f'You ordered food {len(delivery_transactions)} times (avg ₹{avg_delivery_amount:.0f} per order)',
                    'icon': '🛵'
                })
        
        self.insights = insights
        print(f"Generated {len(insights)} insights")
        return insights
        
    def create_visualizations(self, save_plots=True):
            """Create various visualizations"""
            print("Creating visualizations...")
            
            if not hasattr(self, 'summaries'):
                self.calculate_summaries()
            
            plt.style.use('default')
            sns.set_palette("husl")
            
            fig = plt.figure(figsize=(16, 12))
            
            
            ax1 = plt.subplot(2, 3, 1)
            category_data = list(self.summaries['category_spending'].values())
            category_labels = list(self.summaries['category_spending'].keys())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(category_labels)))
            wedges, texts, autotexts = ax1.pie(category_data, labels=category_labels, autopct='%1.1f%%', 
                                            colors=colors, startangle=90)
            ax1.set_title('Spending by Category', fontsize=14, fontweight='bold')
            
            ax2 = plt.subplot(2, 3, 2)
            monthly_data = self.summaries['monthly_spending']
            months = list(monthly_data.keys())
            amounts = list(monthly_data.values())
            
            ax2.plot(months, amounts, marker='o', linewidth=2, markersize=6)
            ax2.set_title('Monthly Spending Trend', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Amount (₹)')
            ax2.tick_params(axis='x', rotation=45)
            
            ax3 = plt.subplot(2, 3, 3)
            daily_data = self.data.groupby(self.data['date'].dt.date)['amount'].sum()
            ax3.bar(range(len(daily_data)), daily_data.values, alpha=0.7)
            ax3.set_title('Daily Spending Pattern', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Days')
            ax3.set_ylabel('Amount (₹)')
            
            ax4 = plt.subplot(2, 3, 4)
            transaction_counts = list(self.summaries['transactions_per_category'].values())
            category_names = list(self.summaries['transactions_per_category'].keys())
            
            bars = ax4.bar(category_names, transaction_counts, color=colors[:len(category_names)])
            ax4.set_title('Transactions per Category', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Category')
            ax4.set_ylabel('Number of Transactions')
            ax4.tick_params(axis='x', rotation=45)
            
            ax5 = plt.subplot(2, 3, 5)
            weekly_data = self.summaries['weekly_spending']
            weeks = list(weekly_data.keys())
            weekly_amounts = list(weekly_data.values())
            
            ax5.plot(weeks, weekly_amounts, marker='s', linewidth=2, markersize=4, color='green')
            ax5.set_title('Weekly Spending Trend', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Week')
            ax5.set_ylabel('Amount (₹)')
            ax5.tick_params(axis='x', rotation=45)
            
            ax6 = plt.subplot(2, 3, 6)
            top_transactions = self.data.nlargest(10, 'amount')
            
            bars = ax6.barh(range(len(top_transactions)), top_transactions['amount'])
            ax6.set_yticks(range(len(top_transactions)))
            ax6.set_yticklabels([desc[:20] + '...' if len(desc) > 20 else desc 
                                for desc in top_transactions['description']], fontsize=8)
            ax6.set_title('Top 10 Transactions', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Amount (₹)')
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('expense_analysis.png', dpi=300, bbox_inches='tight')
                print("Visualizations saved as 'expense_analysis.png'")
            
            plt.show()
        
    def save_processed_data(self, filename='processed_expenses.json'):
        """Save processed data to JSON file"""
        print(f"Saving processed data to {filename}...")
            
        if not hasattr(self, 'summaries'):
            self.calculate_summaries()
            
        if not hasattr(self, 'insights'):
            self.generate_insights()
            
        processed_data = {
            'metadata': {
                'total_transactions': len(self.data),
                'date_range': self.summaries['date_range'],
                'processed_at': datetime.now().isoformat()
            },
            'summaries': self.summaries,
            'insights': self.insights,
            'transactions': self.data.to_dict('records')
        }
       
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        processed_data = convert_numpy_types(processed_data)
        
        with open(filename, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
        
        print(f"Processed data saved to '{filename}'")
    
    def export_to_csv(self, filename='expense_analysis_report.csv'):
        """Export analysis to CSV"""
        print(f"Exporting analysis report to {filename}...")
        
        if not hasattr(self, 'summaries'):
            self.calculate_summaries()
        
      
        report_data = []
        
    
        for category, amount in self.summaries['category_spending'].items():
            percentage = (amount / self.summaries['total_spending']) * 100
            report_data.append({
                'Type': 'Category Summary',
                'Category': category,
                'Amount': amount,
                'Percentage': f"{percentage:.1f}%",
                'Transactions': self.summaries['transactions_per_category'].get(category, 0)
            })
        
   
        for month, amount in self.summaries['monthly_spending'].items():
            report_data.append({
                'Type': 'Monthly Summary',
                'Category': str(month),
                'Amount': amount,
                'Percentage': '',
                'Transactions': ''
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(filename, index=False)
        print(f"Analysis report exported to '{filename}'")
    
    def run_complete_analysis(self):
        """Run complete expense analysis pipeline"""
        print("FINTWIN EXPENSE ANALYZER")
        print("=" * 50)
        
       
        file_path, file_type = self.get_file_input()
        
        
        if file_type == 'pdf':
            print("\nConverting PDF to CSV...")
            csv_path = self.pdf_to_csv(file_path)
            if not csv_path:
                print("Failed to convert PDF. Please try with a CSV file.")
                return False
            file_path = csv_path
        
       
        if not self.load_csv(file_path):
            return False
        
       
        self.categorize_expenses()
        
        
        summaries = self.calculate_summaries()
        
       
        insights = self.generate_insights()
     
        self._print_analysis_results()
        
       
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            print("   Continuing with other outputs...")
        
       
        self.save_processed_data()
        
      
        self.export_to_csv()
        
        print("\nAnalysis completed successfully!")
        print(" Files created:")
        print("   - processed_expenses.json (Complete analysis data)")
        print("   - expense_analysis_report.csv (Summary report)")
        print("   - expense_analysis.png (Visualizations)")
        
        return True

    def _print_analysis_results(self):
        """Print formatted analysis results"""
        print("\nEXPENSE ANALYSIS RESULTS")
        print("=" * 50)

        if not hasattr(self, 'summaries'):
            self.calculate_summaries()
        if not hasattr(self, 'insights'):
            self.generate_insights()

       
        print(f"\nTotal Spending: ₹{self.summaries['total_spending']:.2f}")

       
        print("\nTop Spending Categories:")
        for category in self.summaries['top_categories']:
            amount = self.summaries['category_spending'][category]
            print(f"   - {category}: ₹{amount:.2f}")

        
        print("\nAverages:")
        print(f"   - Average Daily Spending: ₹{self.summaries['avg_daily_spending']:.2f}")
        print(f"   - Average Transaction Amount: ₹{self.summaries['avg_transaction_amount']:.2f}")

        print(f"\nTotal Transactions: {self.summaries['total_transactions']}")

       
        start = self.summaries['date_range']['start']
        end = self.summaries['date_range']['end']
        days = self.summaries['date_range']['days']
        print(f"   - Date Range: {start} to {end} ({days} days)")

        
        print("\nCategory Breakdown:")
        for category, amount in self.summaries['category_spending'].items():
            percent = (amount / self.summaries['total_spending']) * 100
            print(f"   - {category}: ₹{amount:.2f} ({percent:.1f}%)")

        
        print("\nInsights:")
        for insight in self.insights:
            print(f"   {insight['icon']}  {insight['title']}\n      {insight['message']}")


def main():
    analyzer = ExpenseAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
