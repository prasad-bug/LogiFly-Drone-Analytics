import pandas as pd
import numpy as np
import os

def generate_business_excel():
    output_path = 'LogiFly_Business_Data.xlsx'
    
    # --- 1. Financial Projection Data ---
    ANNUAL_COST = 15000000
    INVESTMENT = 8500000
    REDUCTION_PCT = 0.35
    
    years, savings_list, investments, net_benefits, cumulative = [], [], [], [], []
    cum = 0
    for y in range(1, 6):
        s = ANNUAL_COST * REDUCTION_PCT * (1 + 0.03 * (y - 1))
        i = INVESTMENT if y == 1 else INVESTMENT * 0.05
        nb = s - i
        cum += nb
        years.append(f'Year {y}')
        savings_list.append(round(s))
        investments.append(round(i))
        net_benefits.append(round(nb))
        cumulative.append(round(cum))

    df_roi = pd.DataFrame({
        'Year': years,
        'Annual Savings (INR)': savings_list,
        'Investment/Maintenance (INR)': investments,
        'Net Benefit (INR)': net_benefits,
        'Cumulative Net Benefit (INR)': cumulative
    })

    # --- 2. Warehouse Operational Data ---
    np.random.seed(42)
    n_days = 365
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    improvement = np.linspace(0, 1, n_days)
    
    manual_items = np.random.normal(1800, 200, n_days).astype(int).clip(800, 2500)
    manual_accuracy = np.random.normal(78, 5, n_days).clip(60, 90)
    manual_errors = np.random.normal(12, 3, n_days).clip(3, 25)
    manual_time = np.random.normal(6.5, 0.8, n_days).clip(4, 10)
    
    drone_items = np.random.normal(4500, 100, n_days).astype(int).clip(3800, 5500)
    drone_accuracy = np.clip(np.random.normal(96, 1.5, n_days) + improvement * 1.5, 90, 99.9)
    drone_errors = np.clip(np.random.normal(3, 0.8, n_days) - improvement * 0.5, 0.5, 6)
    drone_time = np.clip(np.random.normal(1.2, 0.15, n_days) - improvement * 0.2, 0.5, 2)
    
    df_wh = pd.DataFrame({
        'Date': dates,
        'Manual Items Scanned': manual_items,
        'Drone Items Scanned': drone_items,
        'Manual Accuracy %': manual_accuracy.round(2),
        'Drone Accuracy %': drone_accuracy.round(2),
        'Manual Error Rate %': manual_errors.round(2),
        'Drone Error Rate %': drone_errors.round(2),
        'Manual Scan Time (Hr)': manual_time.round(2),
        'Drone Scan Time (Hr)': drone_time.round(2)
    })

    # --- 3. CRM Data ---
    n_clients = 120
    sectors = ['E-Commerce','Retail','Manufacturing','Pharma','FMCG']
    crm_rows = []
    for i in range(n_clients):
        sector = np.random.choice(sectors, p=[0.35, 0.25, 0.2, 0.1, 0.1])
        base_clv = np.random.randint(5, 50) * 100000
        error_score = np.random.uniform(0, 1)
        retention_pre = float(np.clip(np.random.normal(0.62, 0.12), 0.35, 0.82))
        retention_post = float(np.clip(retention_pre + np.random.uniform(0.18, 0.28), 0.65, 0.97))
        sat_pre = float(np.clip(np.random.normal(58, 10), 35, 75))
        sat_post = float(np.clip(sat_pre + np.random.normal(28, 6), 75, 99))
        crm_rows.append({
            'Client ID': f'CLT{i+1:03d}',
            'Sector': sector,
            'Base CLV': base_clv,
            'Safety/Error Score': round(error_score, 2),
            'Retention (Pre-Drone)': round(retention_pre, 3),
            'Retention (Post-Drone)': round(retention_post, 3),
            'Satisfaction (Pre)': round(sat_pre, 1),
            'Satisfaction (Post)': round(sat_post, 1),
            'CLV Gain (Estimated)': int(base_clv * (retention_post * 1.15 - retention_pre))
        })
    df_crm = pd.DataFrame(crm_rows)

    # --- 4. KPI Summary ---
    kpi_data = {
        'KPI Metric': [
            'Total Annual Savings',
            'Drone Investment',
            'Year 1 Net Benefit',
            'Payback Period (Months)',
            '5-Year Cumulative Profit',
            'Accuracy Improvement (pp)',
            'Error Rate Reduction (%)',
            'Throughput Increase (x)',
            'Scan Time Reduction (%)'
        ],
        'Value': [
            f"INR {ANNUAL_COST * REDUCTION_PCT:,.0f}",
            f"INR {INVESTMENT:,.0f}",
            f"INR {ANNUAL_COST * REDUCTION_PCT - INVESTMENT:,.0f}",
            f"{INVESTMENT / (ANNUAL_COST * REDUCTION_PCT / 12):.1f}",
            f"INR {cumulative[-1]:,.0f}",
            f"{drone_accuracy.mean() - manual_accuracy.mean():.1f}%",
            f"{(manual_errors.mean() - drone_errors.mean()) / manual_errors.mean() * 100:.1f}%",
            f"{drone_items.mean() / manual_items.mean():.1f}x",
            f"{(manual_time.mean() - drone_time.mean()) / manual_time.mean() * 100:.1f}%"
        ]
    }
    df_kpis = pd.DataFrame(kpi_data)

    # Save to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_kpis.to_excel(writer, sheet_name='KPI Summary', index=False)
        df_roi.to_excel(writer, sheet_name='Financial Projection', index=False)
        df_wh.to_excel(writer, sheet_name='Warehouse Operations', index=False)
        df_crm.to_excel(writer, sheet_name='CRM Analytics', index=False)

    print(f"✅ Excel file '{output_path}' generated successfully!")

if __name__ == "__main__":
    generate_business_excel()
