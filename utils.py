#here will live all the chunking, statistics, embedding code

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import os

def generate_synthetic_transactions(
    num_transactions: int = 10000,
    years: int = 2,
    output_file: str = 'synthetic_transactions.csv',
    combine: bool = True,
    annual_revenue_min: float = 500000,
    annual_revenue_max: float = 1300000
) -> pd.DataFrame:
    """
    Generate synthetic business transactions including expenses and Shopify income.

    Args:
        source_csv: Path to your real transaction data
        num_transactions: Number of synthetic expense transactions to generate
        years: How many years of data to span
        output_file: Where to save synthetic data
        combine: Whether to combine with original data
        annual_revenue_min: Minimum annual Shopify revenue (GBP)
        annual_revenue_max: Maximum annual Shopify revenue (GBP)

    Returns:
        DataFrame of synthetic (or combined) transactions
    """

    #df = pd.read_csv(source_csv)
    #print(f"Loaded {len(df)} real transactions")

    # BUSINESS EXPENSE CATEGORIES & MERCHANTS

    business_merchants = {
        'Advertising': [
            'Google Ads', 'Facebook Ads', 'Instagram Ads', 'LinkedIn Ads',
            'Twitter Ads', 'TikTok Ads', 'YouTube Ads', 'Bing Ads'
        ],
        'Software & Subscriptions': [
            'Shopify', 'Slack', 'Notion', 'HubSpot', 'Mailchimp',
            'Canva Pro', 'Adobe Creative Cloud', 'Zoom', 'Dropbox',
            'QuickBooks', 'Xero', 'Asana', 'Monday.com'
        ],
        'Infrastructure & Hosting': [
            'AWS', 'Google Cloud', 'Azure', 'DigitalOcean', 'Heroku',
            'Cloudflare', 'Vercel', 'Netlify', 'MongoDB Atlas'
        ],
        'Inventory & Supplies': [
            'Alibaba', 'Amazon Business', 'Staples', 'Viking Direct',
            'Wholesale Supplier Ltd', 'TradePoint', 'Costco Business'
        ],
        'Employee Salaries': [
            'Payroll - John Smith', 'Payroll - Sarah Jones',
            'Payroll - Marcus Williams', 'Payroll - Priya Patel',
            'Payroll - David Chen', 'Payroll - Emma Thompson'
        ],
        'Owner Salary': [
            'Owner Draw', 'Director Salary', 'Owner Salary Transfer'
        ],
        'Legal & Professional': [
            'Smith & Co Solicitors', 'Legal Zoom', 'Companies House',
            'Trademark Registry', 'Wilson Legal Services'
        ],
        'Accounting & Finance': [
            'Deloitte', 'PwC', 'Local Accountants Ltd', 'Xero Accountant',
            'Bookkeeper Services', 'VAT Returns Ltd', 'HMRC'
        ],
        'Office & Rent': [
            'WeWork', 'Regus', 'Office Space Ltd', 'Business Rates',
            'Commercial Rent', 'Storage Unit', 'Coworking Space'
        ],
        'Travel & Entertainment': [
            'British Airways', 'Trainline', 'Airbnb Business',
            'Client Dinner', 'Uber Business', 'TfL Business'
        ],
        'Insurance': [
            'Hiscox Business Insurance', 'AXA Commercial',
            'Simply Business', 'Direct Line Business'
        ],
        'Banking & Fees': [
            'Stripe Fees', 'PayPal Fees', 'Bank Transfer Fee',
            'Wise Business', 'Square Fees', 'Bank Charges'
        ]
    }

    amount_ranges = {
        'Advertising':              (50,    2000),
        'Software & Subscriptions': (10,    500),
        'Infrastructure & Hosting': (20,    100),
        'Inventory & Supplies':     (100,   10000),
        'Employee Salaries':        (2000,  5000),
        'Owner Salary':             (2000,  5000),
        'Legal & Professional':     (200,   500),
        'Accounting & Finance':     (100,   300),
        'Office & Rent':            (100,   400),
        'Travel & Entertainment':   (20,    800),
        'Insurance':                (50,    500),
        'Banking & Fees':           (1,     100),
    }

    category_weights = {
        'Advertising':              0.05,
        'Software & Subscriptions': 0.10,
        'Infrastructure & Hosting': 0.10,
        'Inventory & Supplies':     0.03,
        'Employee Salaries':        0.08,
        'Owner Salary':             0.04,
        'Legal & Professional':     0.04,
        'Accounting & Finance':     0.04,
        'Office & Rent':            0.06,
        'Travel & Entertainment':   0.01,
        'Insurance':                0.03,
        'Banking & Fees':           0.04,
    }

    categories = list(business_merchants.keys())
    weights = [category_weights[c] for c in categories]

    start_date = datetime.now() - timedelta(days=365 * years)

    # GENERATE EXPENSE TRANSACTIONS

    expense_data = []

    for i in range(num_transactions):
        category = random.choices(categories, weights=weights, k=1)[0]
        merchant = random.choice(business_merchants[category])
        min_amt, max_amt = amount_ranges[category]
        amount = round(np.random.uniform(min_amt, max_amt), 2)
        days_offset = random.randint(0, 365 * years - 1)
        transaction_date = start_date + timedelta(days=days_offset)

        expense_data.append({
            'date': transaction_date.strftime('%Y-%m-%d'),
            'merchant': merchant,
            'amount': -abs(amount),  # Negative = money going out
            'category': category,
            'description': f"{merchant} - {category}",
            'type': 'expense'
        })

    expense_df = pd.DataFrame(expense_data)
    print(f"✓ Generated {len(expense_df)} expense transactions")


    # GENERATE SHOPIFY INCOME TRANSACTIONS
 
    income_data = []

    for year_offset in range(years):
        # Random annual revenue within range
        annual_revenue = random.uniform(annual_revenue_min, annual_revenue_max)

        # Shopify pays out typically daily or weekly - we'll do daily payouts
        # with seasonal variation (Q4 higher due to Christmas, etc.)
        year_start = start_date + timedelta(days=365 * year_offset)

        for day_offset in range(365):
            current_date = year_start + timedelta(days=day_offset)

            # Seasonal multiplier - Q4 boost, Jan slow
            month = current_date.month
            seasonal_multiplier = {
                1: 0.6,   # January - slow
                2: 0.7,
                3: 0.9,
                4: 1.0,
                5: 1.1,
                6: 1.1,
                7: 1.0,
                8: 0.9,
                9: 1.0,
                10: 1.1,
                11: 1.4,  # Black Friday
                12: 1.5   # Christmas
            }.get(month, 1.0)

            # Daily revenue with randomness
            base_daily = annual_revenue / 365
            daily_revenue = base_daily * seasonal_multiplier * np.random.uniform(0.5, 1.5)
            daily_revenue = round(daily_revenue, 2)

            # Skip very low days occasionally (realistic)
            if random.random() < 0.05:
                continue

            income_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'merchant': 'Shopify',
                'amount': abs(daily_revenue),  # Positive = money coming in
                'category': 'Sales Income',
                'description': f'Shopify Daily Payout - Sales Income',
                'type': 'income'
            })

    income_df = pd.DataFrame(income_data)
    print(f"✓ Generated {len(income_df)} Shopify income transactions")
    print(f"  Year 1 revenue: £{income_df[income_df['date'] < (start_date + timedelta(days=365)).strftime('%Y-%m-%d')]['amount'].sum():,.2f}")

   
    # COMBINE EVERYTHING
 
    synthetic_df = pd.concat([expense_df, income_df], ignore_index=True)
    synthetic_df = synthetic_df.sort_values('date').reset_index(drop=True)
    synthetic_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved synthetic data → {output_file}")
    print(f"  Total synthetic transactions: {len(synthetic_df)}")

    # if combine:
    #     # Add type column to original if missing
    #     if 'type' not in df.columns:
    #         df['type'] = 'expense'
    #     combined_df = pd.concat([df, synthetic_df], ignore_index=True)
    #     combined_df = combined_df.sort_values('date').reset_index(drop=True)
    #     combined_df.to_csv('combined_transactions.csv', index=False)
    #     print(f"✓ Combined dataset saved → combined_transactions.csv")
    #     print(f"  Total transactions: {len(combined_df)}")
    #     return combined_df

    return synthetic_df


#Helper function that creates the statistics for the weekly, monthly and quarter chunks from the financial data
def create_period_stats(
    income_df: pd.DataFrame,
    expense_df: pd.DataFrame,
    period,
    period_type :str = 'W',    #'W','M','Q'
    top_n_categories = 3  ) -> dict:
    
    period_col = f'date_period_{period_type}'
    
    period_income = income_df[income_df['date'].dt.to_period(period_type) == period]
    period_expenses = expense_df[expense_df['date'].dt.to_period(period_type) == period]
    
    total_income = period_income['amount'].sum()
    total_expenses = abs(period_expenses['amount'].sum())
    
    net_pl = total_income - total_expenses
    transaction_count =  len(period_income) + len(period_expenses)

    # Top 3 category spends
    top_categories = (
        period_expenses[period_expenses['date'].dt.to_period(period_type) == period]
        .groupby('category')['amount']
        .sum()
        .abs()
        .nlargest(top_n_categories)
    )
    top_cats_text = ', '.join([f"{cat}: £{amt:,.2f}" for cat, amt in top_categories.items()])
    
    return {
        'total_income': total_income,
        'total_expenses' : total_expenses,
        'net_pl' : net_pl,
        'transaction_count': transaction_count,
        'top_cats_text': top_cats_text,
        'top_categories': top_categories
    }   
    
#Create all the chunks: weekly, monthly, quarterly.
def create_chunks(df: pd.DataFrame) -> list[dict]:
    """
    Generate weekly, monthly, and quarterly financial summary chunks
    from a transactions DataFrame.
    
    Each chunk contains:
    - text: human readable summary for embedding
    - metadata: structured data for filtering and retrieval
    """
    
    # Ensure date column is datetime
    #df['date'] = pd.to_datetime(df['date'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    print(f"Clean rows after date parsing: {len(df)}")

    # Separate income and expenses
    income_df = df[df['type'] == 'income'].copy()
    expense_df = df[df['type'] == 'expense'].copy()
    
    chunks = []


    # WEEKLY CHUNKS
  
    df['week'] = df['date'].dt.to_period('W')
    weekly_groups = df.groupby('week')
    
    prev_week_pl = None
    
    for week, group in weekly_groups:
        
        weekly_stats = create_period_stats(income_df,expense_df,week,'W',3)
        
        # Week over week delta
        wow_delta = weekly_stats['net_pl'] - prev_week_pl if prev_week_pl is not None else 0
        wow_text = f"+£{wow_delta:,.2f}" if wow_delta >= 0 else f"-£{abs(wow_delta):,.2f}"
        
        chunk_text = f"""
        Weekly Financial Summary
        Period: {week}
        Total Income: £{weekly_stats['total_income']:,.2f}
        Total Expenses: £{weekly_stats['total_expenses']:,.2f}
        Net P&L: £{weekly_stats['net_pl']:,.2f}
        Week-over-Week Change: {wow_text}
        Top Spending Categories: {weekly_stats['top_cats_text']}
        Transaction Count: {weekly_stats['transaction_count']}
        """
        chunks.append({
            'text': chunk_text,
            'metadata': {
                'chunk_type': 'weekly',
                'period': str(week),
                'total_income': round(weekly_stats['total_income'], 2),
                'total_expenses': round(weekly_stats['total_expenses'], 2),
                'net_pl': round(weekly_stats['net_pl'], 2),
                'transaction_count': weekly_stats['transaction_count']
            }
        })
        
        prev_week_pl = weekly_stats['net_pl']

    print(f"✓ Created {len([c for c in chunks if c['metadata']['chunk_type'] == 'weekly'])} weekly chunks")

    
    # MONTHLY CHUNKS
    
    
    df['month'] = df['date'].dt.to_period('M')
    monthly_groups = df.groupby('month')
    
    monthly_pls = {}  # Store for quarterly use and trend
    
    for month, group in monthly_groups:

        monthly_stats = create_period_stats(income_df,expense_df,month,'M',5)
        
        # Employee salary total
        employee_spend = abs(
            expense_df[
                (expense_df['date'].dt.to_period('M') == month) &
                (expense_df['category'] == 'Employee Salaries')
            ]['amount'].sum()
        )
        
        monthly_pls[month] = monthly_stats['net_pl']
        
        # 3 month cashflow trend
        sorted_months = sorted(monthly_pls.keys())
        recent_months = sorted_months[-3:]
        trend_text = ' → '.join([f"£{monthly_pls[m]:,.2f}" for m in recent_months])
        
        chunk_text = f"""
        Monthly Financial Summary
        Period: {month}
        Total Income: £{monthly_stats['total_income']:,.2f}
        Total Expenses: £{monthly_stats['total_expenses']:,.2f}
        Net P&L: £{monthly_stats['net_pl']:,.2f}
        Employee Salary Total: £{employee_spend:,.2f}
        Top Spending Categories: {monthly_stats['top_cats_text']}
        3-Month Cashflow Trend: {trend_text}
        Transaction Count: {monthly_stats['transaction_count']}
        """
        chunks.append({
            'text': chunk_text,
            'metadata': {
                'chunk_type': 'monthly',
                'period': str(month),
                'total_income': round(monthly_stats['total_income'], 2),
                'total_expenses': round(monthly_stats['total_expenses'], 2),
                'net_pl': round(monthly_stats['net_pl'], 2),
                'employee_spend': round(employee_spend, 2),
                'transaction_count': monthly_stats['transaction_count']
            }
        })

    print(f"✓ Created {len([c for c in chunks if c['metadata']['chunk_type'] == 'monthly'])} monthly chunks")

    
    # QUARTERLY CHUNKS
    
    df['quarter'] = df['date'].dt.to_period('Q')
    quarterly_groups = df.groupby('quarter')
    
    prev_quarter_pl = None
    
    for quarter, group in quarterly_groups:

        quarter_stats = create_period_stats(income_df, expense_df, quarter, 'Q', 5)
        # Best performing month in quarter
        quarter_months = [m for m in monthly_pls.keys() if m.year == quarter.year and ((m.month - 1) // 3 + 1) == quarter.quarter]
        if quarter_months:
            best_month = max(quarter_months, key=lambda m: monthly_pls[m])
            best_month_text = f"{best_month} (£{monthly_pls[best_month]:,.2f})"
        else:
            best_month_text = "N/A"
        
        # Quarter over quarter delta
        qoq_delta = quarter_stats['net_pl'] - prev_quarter_pl if prev_quarter_pl is not None else 0
        qoq_text = f"+£{qoq_delta:,.2f}" if qoq_delta >= 0 else f"-£{abs(qoq_delta):,.2f}"
        
        chunk_text = f"""
        Quarterly Financial Summary
        Period: {quarter}
        Total Income: £{quarter_stats['total_income']:,.2f}
        Total Expenses: £{quarter_stats['total_expenses']:,.2f}
        Net P&L: £{quarter_stats['net_pl']:,.2f}
        Quarter-over-Quarter Change: {qoq_text}
        Best Performing Month: {best_month_text}
        Top Spending Categories: {quarter_stats['top_cats_text']}
        Transaction Count: {quarter_stats['transaction_count']}
        """
        chunks.append({
            'text': chunk_text,
            'metadata': {
                'chunk_type': 'quarterly',
                'period': str(quarter),
                'total_income': round(quarter_stats['total_income'], 2),
                'total_expenses': round(quarter_stats['total_expenses'], 2),
                'net_pl': round(quarter_stats['net_pl'], 2),
                'best_month': best_month_text,
                'transaction_count': quarter_stats['transaction_count']
            }
        })
        
        prev_quarter_pl = quarter_stats['net_pl']

    print(f"✓ Created {len([c for c in chunks if c['metadata']['chunk_type'] == 'quarterly'])} quarterly chunks")
    print(f"✓ Total chunks created: {len(chunks)}")
    
    return chunks


def embed_and_store(chunks: list[dict], collection_name: str = 'financial_data'):
    """
    Embed all chunks and store in Chroma vector DB with metadata.
    """
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    client_db = chromadb.PersistentClient(path='./chroma') #chromadb.Client()
    collection = client_db.get_or_create_collection(name=collection_name) #create_collection(name=collection_name)
    
    # If collection already has data, skip re-embedding
    if collection.count() > 0:
        print(f"✓ Collection already exists with {collection.count()} chunks - skipping embedding")
        return collection, embedding_model
    
    # Otherwise embed and store
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk['text'])
        collection.add(
            ids=[str(i)],
            embeddings=[embedding.tolist()],
            documents=[chunk['text']],
            metadatas=[chunk['metadata']]
        )
    
    print(f"✓ Stored {len(chunks)} chunks in vector database")
    return collection, embedding_model

#the orchestration happens here
def ingest_data(csv_path: str = 'synthetic_transactions.csv'):
    
    # Generate synthetic data if CSV doesn't exist
    if not os.path.exists(csv_path):
        print("No CSV found - generating synthetic transactions...")
        generate_synthetic_transactions(
            num_transactions=10000,
            years=2,
            output_file=csv_path,
            combine=False  # no real data to combine with
        )
    
    print("Loading data...")
    df = pd.read_csv(csv_path,on_bad_lines='skip')
    
    print("Creating chunks...")
    chunks = create_chunks(df)
    
    print("Embedding and storing...")
    collection, embedding_model = embed_and_store(chunks)
    
    print("✓ Ingestion complete")
    return collection, embedding_model

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('synthetic_transactions.csv')
    chunks = create_chunks(df)
    collection, embedding_model = embed_and_store(chunks)
    print("utils.py ✓")