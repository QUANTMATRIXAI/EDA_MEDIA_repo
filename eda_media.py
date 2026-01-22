import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Marketing Analytics Dashboard", layout="wide")
st.title("Marketing Analytics Dashboard")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = {'meta': None, 'shopify': None, 'ga': None}

# File upload section
st.header("1. Upload Data Sources")
col1, col2, col3 = st.columns(3)

with col1:
    meta_file = st.file_uploader("Upload Meta Data", type=['csv', 'xlsx', 'parquet'])
    if meta_file:
        if meta_file.name.endswith('.parquet'):
            st.session_state.data['meta'] = pd.read_parquet(meta_file)
        elif meta_file.name.endswith('.csv'):
            st.session_state.data['meta'] = pd.read_csv(meta_file)
        else:
            st.session_state.data['meta'] = pd.read_excel(meta_file)
        st.success(f"Meta: {len(st.session_state.data['meta'])} rows")

with col2:
    shopify_file = st.file_uploader("Upload Shopify Data", type=['csv', 'xlsx', 'parquet'])
    if shopify_file:
        if shopify_file.name.endswith('.parquet'):
            st.session_state.data['shopify'] = pd.read_parquet(shopify_file)
        elif shopify_file.name.endswith('.csv'):
            st.session_state.data['shopify'] = pd.read_csv(shopify_file)
        else:
            st.session_state.data['shopify'] = pd.read_excel(shopify_file)
        st.success(f"Shopify: {len(st.session_state.data['shopify'])} rows")

with col3:
    ga_file = st.file_uploader("Upload GA Data", type=['csv', 'xlsx', 'parquet'])
    if ga_file:
        if ga_file.name.endswith('.parquet'):
            st.session_state.data['ga'] = pd.read_parquet(ga_file)
        elif ga_file.name.endswith('.csv'):
            st.session_state.data['ga'] = pd.read_csv(ga_file)
        else:
            st.session_state.data['ga'] = pd.read_excel(ga_file)
        st.success(f"GA: {len(st.session_state.data['ga'])} rows")

# Check if all data is loaded
if all(v is not None for v in st.session_state.data.values()):
    
    # Auto-detect date columns
    def detect_date_column(df):
        for col in df.columns:
            if col.lower() in ['date', 'day']:
                return col
        return None
    
    date_cols = {
        'meta': detect_date_column(st.session_state.data['meta']),
        'shopify': detect_date_column(st.session_state.data['shopify']),
        'ga': detect_date_column(st.session_state.data['ga'])
    }
    
    st.header("2. Select Periods and Filters")
    
    col1, col2 = st.columns(2)
    with col1:
        campaign_start = st.date_input("Campaign Period Start", value=datetime(2025, 8, 14))
        campaign_end = st.date_input("Campaign Period End", value=datetime(2025, 9, 10))
    
    with col2:
        pre_campaign_start = st.date_input("Pre-Campaign Period Start", value=datetime(2025, 7, 17))
        pre_campaign_end = st.date_input("Pre-Campaign Period End", value=datetime(2025, 8, 13))
    
    # Region column selection
    st.subheader("Select Region Column for Each Data Source")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        meta_region_col = st.selectbox("Meta region column", 
            ['None'] + list(st.session_state.data['meta'].columns),
            index=0)
    
    with col2:
        shopify_region_col = st.selectbox("Shopify region column", 
            ['None'] + list(st.session_state.data['shopify'].columns),
            index=0)
    
    with col3:
        ga_region_col = st.selectbox("GA region column", 
            ['None'] + list(st.session_state.data['ga'].columns),
            index=0)
    
    region_cols = {
        'meta': meta_region_col if meta_region_col != 'None' else None,
        'shopify': shopify_region_col if shopify_region_col != 'None' else None,
        'ga': ga_region_col if ga_region_col != 'None' else None
    }
    
    # Region selection
    all_regions = []
    if region_cols['shopify']:
        all_regions += list(st.session_state.data['shopify'][region_cols['shopify']].unique())
    if region_cols['ga']:
        all_regions += list(st.session_state.data['ga'][region_cols['ga']].unique())
    if region_cols['meta']:
        all_regions += list(st.session_state.data['meta'][region_cols['meta']].unique())
    
    all_regions = list(set(all_regions))
    
    selected_regions = st.multiselect("Select Regions (optional)", all_regions, default=all_regions if all_regions else None)
    
    # Meta/Non-Meta split column selection
    st.subheader("Meta/Non-Meta Attribution Column (optional)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        meta_split_col = st.selectbox("Meta split column", 
            ['None'] + list(st.session_state.data['meta'].columns),
            index=0)
    
    with col2:
        shopify_split_col = st.selectbox("Shopify split column", 
            ['None'] + list(st.session_state.data['shopify'].columns),
            index=0)
    
    with col3:
        ga_split_col = st.selectbox("GA split column", 
            ['None'] + list(st.session_state.data['ga'].columns),
            index=0)
    
    st.header("3. Select Metrics")
    
    # Get numeric/metric columns only
    def get_metric_columns(df, exclude_cols):
        return [col for col in df.columns 
                if col not in exclude_cols 
                and (df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df[col]))]
    
    meta_exclude = [date_cols['meta'], region_cols['meta']] + ([meta_split_col] if meta_split_col != 'None' else [])
    meta_exclude = [col for col in meta_exclude if col]
    
    shopify_exclude = [date_cols['shopify'], region_cols['shopify']] + ([shopify_split_col] if shopify_split_col != 'None' else [])
    shopify_exclude = [col for col in shopify_exclude if col]
    
    ga_exclude = [date_cols['ga'], region_cols['ga']] + ([ga_split_col] if ga_split_col != 'None' else [])
    ga_exclude = [col for col in ga_exclude if col]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Meta Metrics")
        meta_available = get_metric_columns(st.session_state.data['meta'], meta_exclude)
        meta_metrics = st.multiselect("Meta", meta_available,
            default=meta_available[:2] if len(meta_available) >= 2 else meta_available)
    
    with col2:
        st.subheader("Shopify Metrics")
        shopify_available = get_metric_columns(st.session_state.data['shopify'], shopify_exclude)
        shopify_metrics = st.multiselect("Shopify", shopify_available,
            default=shopify_available[:2] if len(shopify_available) >= 2 else shopify_available)
    
    with col3:
        st.subheader("GA Metrics")
        ga_available = get_metric_columns(st.session_state.data['ga'], ga_exclude)
        
        # Set default base metrics - the 3 main ones
        ga_default_base = []
        for possible_names in [['Sessions', 'sessions', 'Total Sessions'], 
                               ['Total users', 'Total Users', 'total_users'],
                               ['Total purchasers', 'Total Purchasers', 'total_purchasers']]:
            for name in possible_names:
                if name in ga_available and name not in ga_default_base:
                    ga_default_base.append(name)
                    break
        
        # If we didn't find them, just take first 3
        if len(ga_default_base) < 3:
            ga_default_base = ga_available[:3] if len(ga_available) >= 3 else ga_available
        
        ga_metrics = st.multiselect("GA Base Metrics", ga_available, default=ga_default_base)
        
        # Initialize default ratios if not exists
        if 'ga_custom_ratios' not in st.session_state:
            st.session_state.ga_custom_ratios = []
            
            # Find the exact columns from your data - exact case matching
            sessions_col = None
            users_col = None
            purchasers_col = None
            engaged_sessions = None
            new_users = None
            items_viewed = None
            add_to_carts = None
            avg_session_duration = None
            
            for col in ga_available:
                if col == 'Sessions':
                    sessions_col = col
                elif col == 'Total users':
                    users_col = col
                elif col == 'Total purchasers':
                    purchasers_col = col
                elif col == 'Engaged sessions':
                    engaged_sessions = col
                elif col == 'New users':
                    new_users = col
                elif col == 'Items viewed':
                    items_viewed = col
                elif col == 'Add to carts':
                    add_to_carts = col
                elif col == 'Average session duration':
                    avg_session_duration = col
            
            # Create the exact default ratios
            if engaged_sessions and sessions_col:
                st.session_state.ga_custom_ratios.append(f"{engaged_sessions}/{sessions_col}")
            if new_users and users_col:
                st.session_state.ga_custom_ratios.append(f"{new_users}/{users_col}")
            if items_viewed and users_col:
                st.session_state.ga_custom_ratios.append(f"{items_viewed}/{users_col}")
            if add_to_carts and sessions_col:
                st.session_state.ga_custom_ratios.append(f"{add_to_carts}/{sessions_col}")
            if purchasers_col and users_col:
                st.session_state.ga_custom_ratios.append(f"{purchasers_col}/{users_col}")
    
    # GA Ratio Builder
    with st.expander("➗ Manage GA Ratios"):
        st.markdown("**Build custom ratio metrics (numerator / denominator)**")
        st.markdown("*Ratios will be calculated as: SUM(numerator) / SUM(denominator)*")
        
        # Reset to defaults button
        if st.button("🔄 Reset to Default Ratios", key='reset_ratios'):
            st.session_state.ga_custom_ratios = []
            
            # Find the exact columns from your data - exact case matching
            sessions_col = None
            users_col = None
            purchasers_col = None
            engaged_sessions = None
            new_users = None
            items_viewed = None
            add_to_carts = None
            
            for col in ga_available:
                if col == 'Sessions':
                    sessions_col = col
                elif col == 'Total users':
                    users_col = col
                elif col == 'Total purchasers':
                    purchasers_col = col
                elif col == 'Engaged sessions':
                    engaged_sessions = col
                elif col == 'New users':
                    new_users = col
                elif col == 'Items viewed':
                    items_viewed = col
                elif col == 'Add to carts':
                    add_to_carts = col
            
            # Create the exact default ratios
            if engaged_sessions and sessions_col:
                st.session_state.ga_custom_ratios.append(f"{engaged_sessions}/{sessions_col}")
            if new_users and users_col:
                st.session_state.ga_custom_ratios.append(f"{new_users}/{users_col}")
            if items_viewed and users_col:
                st.session_state.ga_custom_ratios.append(f"{items_viewed}/{users_col}")
            if add_to_carts and sessions_col:
                st.session_state.ga_custom_ratios.append(f"{add_to_carts}/{sessions_col}")
            if purchasers_col and users_col:
                st.session_state.ga_custom_ratios.append(f"{purchasers_col}/{users_col}")
            
            st.success(f"Reset to {len(st.session_state.ga_custom_ratios)} default ratios!")
            st.rerun()
        
        # Show current ratios first
        if st.session_state.ga_custom_ratios:
            st.markdown("**Current Ratios:**")
            for idx, ratio in enumerate(st.session_state.ga_custom_ratios):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(ratio)
                with col2:
                    if st.button("🗑️", key=f'remove_ratio_{idx}'):
                        st.session_state.ga_custom_ratios.remove(ratio)
                        st.rerun()
            st.markdown("---")
        else:
            st.info("No ratios defined yet. Click 'Reset to Default Ratios' or add one below!")
        
        st.markdown("**Add New Ratio:**")
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            ga_num = st.selectbox("Numerator", ['Select...'] + ga_available, key='ga_ratio_num')
        
        with col2:
            ga_den = st.selectbox("Denominator", ['Select...'] + ga_available, key='ga_ratio_den')
        
        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("➕ Add Ratio", key='add_ga_ratio'):
                if ga_num != 'Select...' and ga_den != 'Select...':
                    ratio_name = f"{ga_num}/{ga_den}"
                    if ratio_name not in st.session_state.ga_custom_ratios:
                        st.session_state.ga_custom_ratios.append(ratio_name)
                        st.success(f"Added: {ratio_name}")
                        st.rerun()
                    else:
                        st.warning("Ratio already exists")
        
        # Add ratios to ga_metrics
        ga_metrics = list(ga_metrics) + st.session_state.ga_custom_ratios
    
    # Pivot options
    st.header("4. Pivot Options")
    
    # Get non-metric columns for pivot
    meta_pivot_cols = [col for col in st.session_state.data['meta'].columns if col not in meta_exclude + meta_metrics]
    shopify_pivot_cols = [col for col in st.session_state.data['shopify'].columns if col not in shopify_exclude + shopify_metrics]
    ga_pivot_cols = [col for col in st.session_state.data['ga'].columns if col not in ga_exclude + ga_metrics]
    
    pivot_options = ['None']
    if meta_pivot_cols:
        pivot_options += [f"Meta: {col}" for col in meta_pivot_cols]
    if shopify_pivot_cols:
        pivot_options += [f"Shopify: {col}" for col in shopify_pivot_cols]
    if ga_pivot_cols:
        pivot_options += [f"GA: {col}" for col in ga_pivot_cols]
    
    pivot_selection = st.selectbox("Group by column (optional)", pivot_options)
    
    if st.button("Generate Report", type="primary"):
        
        # Parse pivot selection
        pivot_col = None
        pivot_source = None
        if pivot_selection != 'None':
            parts = pivot_selection.split(': ')
            if len(parts) == 2:
                pivot_source = parts[0]
                pivot_col = parts[1]
        
        def normalize_group_value(val):
            return str(val).strip().lower()

        def normalize_split_value(val):
            if val is None:
                return None
            text = str(val).strip().lower()
            text = text.replace('_', ' ').replace('-', ' ')
            text = ' '.join(text.split())
            if 'non meta' in text or text.startswith('non'):
                return 'non-meta'
            if 'meta' in text:
                return 'meta'
            return text

        def make_key(split_val, pivot_val, metric):
            split_norm = normalize_group_value(split_val) if split_val is not None else None
            pivot_norm = normalize_group_value(pivot_val) if pivot_val is not None else None
            return (split_norm, pivot_norm, metric)

        def build_index(data_dict):
            idx = {}
            for (split_val, pivot_val, metric), value in data_dict.items():
                split_key = normalize_split_value(split_val) if split_val is not None else None
                idx.setdefault(metric, {}).setdefault(pivot_val, {}).setdefault(split_key, 0)
                idx[metric][pivot_val][split_key] += value
            return idx

        def get_index_value(index, metric, pivot_val=None, split_key=None):
            return index.get(metric, {}).get(pivot_val, {}).get(split_key, 0)

        # Filter and aggregate function
        def process_data(df, metrics, start_date, end_date, regions, date_col, region_col, split_col, pivot_col=None):
            # Filter by date
            filtered = df.copy()
            if date_col:
                filtered[date_col] = pd.to_datetime(filtered[date_col])
                filtered = filtered[(filtered[date_col] >= pd.to_datetime(start_date)) & 
                                  (filtered[date_col] <= pd.to_datetime(end_date))]
            
            # Filter by region
            if region_col and regions:
                filtered = filtered[filtered[region_col].isin(regions)]
            
            results = {}
            
            # Group by split and/or pivot column
            if split_col and split_col != 'None' and split_col in filtered.columns:
                if pivot_col and pivot_col in filtered.columns:
                    # Both split and pivot
                    for split_val in filtered[split_col].unique():
                        split_df = filtered[filtered[split_col] == split_val]
                        for pivot_val in split_df[pivot_col].unique():
                            pivot_df = split_df[split_df[pivot_col] == pivot_val]
                            for metric in metrics:
                                if metric in pivot_df.columns:
                                    key = make_key(split_val, pivot_val, metric)
                                    results[key] = pivot_df[metric].sum()
                else:
                    # Only split
                    for split_val in filtered[split_col].unique():
                        split_df = filtered[filtered[split_col] == split_val]
                        for metric in metrics:
                            if metric in split_df.columns:
                                key = make_key(split_val, None, metric)
                                results[key] = split_df[metric].sum()
            elif pivot_col and pivot_col in filtered.columns:
                # Only pivot
                for pivot_val in filtered[pivot_col].unique():
                    pivot_df = filtered[filtered[pivot_col] == pivot_val]
                    for metric in metrics:
                        if metric in pivot_df.columns:
                            key = make_key(None, pivot_val, metric)
                            results[key] = pivot_df[metric].sum()
            else:
                # No grouping
                for metric in metrics:
                    if metric in filtered.columns:
                        key = make_key(None, None, metric)
                        results[key] = filtered[metric].sum()
            
            return results, filtered

        def find_sessions_column(df):
            candidates = {'sessions', 'total sessions'}
            for col in df.columns:
                norm = str(col).strip().lower().replace('_', ' ')
                if norm in candidates:
                    return col
            return None

        def process_weighted_average(df, metric, weight_col, start_date, end_date, regions, date_col, region_col, split_col, pivot_col=None):
            filtered = df.copy()
            if date_col:
                filtered[date_col] = pd.to_datetime(filtered[date_col])
                filtered = filtered[(filtered[date_col] >= pd.to_datetime(start_date)) & 
                                  (filtered[date_col] <= pd.to_datetime(end_date))]

            if region_col and regions:
                filtered = filtered[filtered[region_col].isin(regions)]

            results = {}

            def calc_weighted_avg(sub_df):
                if metric not in sub_df.columns:
                    return 0
                values = pd.to_numeric(sub_df[metric], errors='coerce').fillna(0)
                if weight_col and weight_col in sub_df.columns:
                    weights = pd.to_numeric(sub_df[weight_col], errors='coerce').fillna(0)
                    total_weight = weights.sum()
                    if total_weight == 0:
                        return 0
                    return (values * weights).sum() / total_weight
                if values.dropna().empty:
                    return 0
                return values.mean()

            if split_col and split_col != 'None' and split_col in filtered.columns:
                if pivot_col and pivot_col in filtered.columns:
                    for split_val in filtered[split_col].unique():
                        split_df = filtered[filtered[split_col] == split_val]
                        for pivot_val in split_df[pivot_col].unique():
                            pivot_df = split_df[split_df[pivot_col] == pivot_val]
                            key = make_key(split_val, pivot_val, metric)
                            results[key] = calc_weighted_avg(pivot_df)
                else:
                    for split_val in filtered[split_col].unique():
                        split_df = filtered[filtered[split_col] == split_val]
                        key = make_key(split_val, None, metric)
                        results[key] = calc_weighted_avg(split_df)
            elif pivot_col and pivot_col in filtered.columns:
                for pivot_val in filtered[pivot_col].unique():
                    pivot_df = filtered[filtered[pivot_col] == pivot_val]
                    key = make_key(None, pivot_val, metric)
                    results[key] = calc_weighted_avg(pivot_df)
            else:
                key = make_key(None, None, metric)
                results[key] = calc_weighted_avg(filtered)

            return results, filtered
        
        # Process all data
        region_divisor = len(selected_regions) if len(selected_regions) > 1 else 1
        results = []
        
        # Store filtered data for debug
        filtered_data = {
            'meta_campaign': None,
            'meta_pre': None,
            'shopify_campaign': None,
            'shopify_pre': None,
            'ga_campaign': None,
            'ga_pre': None
        }
        
        # Meta
        for metric in meta_metrics:
            use_pivot = pivot_col if pivot_source == 'Meta' else None
            meta_camp, filtered_data['meta_campaign'] = process_data(st.session_state.data['meta'], [metric], campaign_start, campaign_end,
                                    selected_regions, date_cols['meta'], region_cols['meta'], meta_split_col, use_pivot)
            meta_pre, filtered_data['meta_pre'] = process_data(st.session_state.data['meta'], [metric], pre_campaign_start, pre_campaign_end,
                                   selected_regions, date_cols['meta'], region_cols['meta'], meta_split_col, use_pivot)
            meta_camp_index = build_index(meta_camp)
            meta_pre_index = build_index(meta_pre)
            
            if use_pivot:
                # Extract pivot values
                pivot_values = set()
                for (_, pivot_val, _) in list(meta_camp.keys()) + list(meta_pre.keys()):
                    if pivot_val is not None:
                        pivot_values.add(pivot_val)
                
                for piv_val in pivot_values:
                    if meta_split_col != 'None':
                        row = {
                            'Data source': 'Meta',
                            'Metric': f"{piv_val}_{metric}",
                            'Meta (Pre-Campaign)': 0,
                            'Non-meta (Pre-Campaign)': 0,
                            'Meta (Campaign)': 0,
                            'Non-meta (Campaign)': 0
                        }
                        row['Meta (Campaign)'] = get_index_value(meta_camp_index, metric, piv_val, 'meta') / region_divisor
                        row['Non-meta (Campaign)'] = get_index_value(meta_camp_index, metric, piv_val, 'non-meta') / region_divisor
                        row['Meta (Pre-Campaign)'] = get_index_value(meta_pre_index, metric, piv_val, 'meta') / region_divisor
                        row['Non-meta (Pre-Campaign)'] = get_index_value(meta_pre_index, metric, piv_val, 'non-meta') / region_divisor
                    else:
                        row = {
                            'Data source': 'Meta',
                            'Metric': f"{piv_val}_{metric}",
                            'Meta (Pre-Campaign)': get_index_value(meta_pre_index, metric, piv_val, None) / region_divisor,
                            'Non-meta (Pre-Campaign)': '',
                            'Meta (Campaign)': get_index_value(meta_camp_index, metric, piv_val, None) / region_divisor,
                            'Non-meta (Campaign)': ''
                        }
                    results.append(row)
            elif meta_split_col != 'None':
                row = {
                    'Data source': 'Meta',
                    'Metric': metric,
                    'Meta (Pre-Campaign)': get_index_value(meta_pre_index, metric, None, 'meta') / region_divisor,
                    'Non-meta (Pre-Campaign)': get_index_value(meta_pre_index, metric, None, 'non-meta') / region_divisor,
                    'Meta (Campaign)': get_index_value(meta_camp_index, metric, None, 'meta') / region_divisor,
                    'Non-meta (Campaign)': get_index_value(meta_camp_index, metric, None, 'non-meta') / region_divisor
                }
            else:
                row = {
                    'Data source': 'Meta',
                    'Metric': metric,
                    'Meta (Pre-Campaign)': get_index_value(meta_pre_index, metric, None, None) / region_divisor,
                    'Non-meta (Pre-Campaign)': '',
                    'Meta (Campaign)': get_index_value(meta_camp_index, metric, None, None) / region_divisor,
                    'Non-meta (Campaign)': ''
                }
            results.append(row)
        
        # Shopify
        for metric in shopify_metrics:
            use_pivot = pivot_col if pivot_source == 'Shopify' else None
            shopify_camp, filtered_data['shopify_campaign'] = process_data(st.session_state.data['shopify'], [metric], campaign_start, campaign_end,
                                       selected_regions, date_cols['shopify'], region_cols['shopify'], shopify_split_col, use_pivot)
            shopify_pre, filtered_data['shopify_pre'] = process_data(st.session_state.data['shopify'], [metric], pre_campaign_start, pre_campaign_end,
                                      selected_regions, date_cols['shopify'], region_cols['shopify'], shopify_split_col, use_pivot)
            shopify_camp_index = build_index(shopify_camp)
            shopify_pre_index = build_index(shopify_pre)
            
            if use_pivot:
                # Extract unique pivot values from keys
                pivot_values = set()
                for (_, pivot_val, _) in list(shopify_camp.keys()) + list(shopify_pre.keys()):
                    if pivot_val is not None:
                        pivot_values.add(pivot_val)
                
                for piv_val in pivot_values:
                    if shopify_split_col != 'None':
                        row = {
                            'Data source': 'Shopify',
                            'Metric': f"{piv_val}_{metric}",
                            'Meta (Pre-Campaign)': 0,
                            'Non-meta (Pre-Campaign)': 0,
                            'Meta (Campaign)': 0,
                            'Non-meta (Campaign)': 0
                        }
                        row['Meta (Campaign)'] = get_index_value(shopify_camp_index, metric, piv_val, 'meta') / region_divisor
                        row['Non-meta (Campaign)'] = get_index_value(shopify_camp_index, metric, piv_val, 'non-meta') / region_divisor
                        row['Meta (Pre-Campaign)'] = get_index_value(shopify_pre_index, metric, piv_val, 'meta') / region_divisor
                        row['Non-meta (Pre-Campaign)'] = get_index_value(shopify_pre_index, metric, piv_val, 'non-meta') / region_divisor
                    else:
                        row = {
                            'Data source': 'Shopify',
                            'Metric': f"{piv_val}_{metric}",
                            'Meta (Pre-Campaign)': get_index_value(shopify_pre_index, metric, piv_val, None) / region_divisor,
                            'Non-meta (Pre-Campaign)': '',
                            'Meta (Campaign)': get_index_value(shopify_camp_index, metric, piv_val, None) / region_divisor,
                            'Non-meta (Campaign)': ''
                        }
                    
                    results.append(row)
            elif shopify_split_col != 'None':
                row = {
                    'Data source': 'Shopify',
                    'Metric': metric,
                    'Meta (Pre-Campaign)': get_index_value(shopify_pre_index, metric, None, 'meta') / region_divisor,
                    'Non-meta (Pre-Campaign)': get_index_value(shopify_pre_index, metric, None, 'non-meta') / region_divisor,
                    'Meta (Campaign)': get_index_value(shopify_camp_index, metric, None, 'meta') / region_divisor,
                    'Non-meta (Campaign)': get_index_value(shopify_camp_index, metric, None, 'non-meta') / region_divisor
                }
            else:
                row = {
                    'Data source': 'Shopify',
                    'Metric': metric,
                    'Meta (Pre-Campaign)': get_index_value(shopify_pre_index, metric, None, None) / region_divisor,
                    'Non-meta (Pre-Campaign)': '',
                    'Meta (Campaign)': get_index_value(shopify_camp_index, metric, None, None) / region_divisor,
                    'Non-meta (Campaign)': ''
                }
            results.append(row)
        
        # GA
        ga_sessions_col = find_sessions_column(st.session_state.data['ga'])
        for metric in ga_metrics:
            use_pivot = pivot_col if pivot_source == 'GA' else None
            metric_norm = str(metric).strip().lower().replace('.', '')
            is_avg_session_duration = ('average session duration' in metric_norm or
                                       'avg session duration' in metric_norm)
            
            # Check if it's a custom ratio
            if '/' in metric:
                parts = metric.split('/')
                if len(parts) == 2:
                    num_metric = parts[0].strip()
                    den_metric = parts[1].strip()
                    
                    # Get numerator and denominator data
                    ga_camp_num, _ = process_data(st.session_state.data['ga'], [num_metric], campaign_start, campaign_end,
                                                  selected_regions, date_cols['ga'], region_cols['ga'], ga_split_col, use_pivot)
                    ga_pre_num, _ = process_data(st.session_state.data['ga'], [num_metric], pre_campaign_start, pre_campaign_end,
                                                 selected_regions, date_cols['ga'], region_cols['ga'], ga_split_col, use_pivot)
                    
                    ga_camp_den, _ = process_data(st.session_state.data['ga'], [den_metric], campaign_start, campaign_end,
                                                  selected_regions, date_cols['ga'], region_cols['ga'], ga_split_col, use_pivot)
                    ga_pre_den, _ = process_data(st.session_state.data['ga'], [den_metric], pre_campaign_start, pre_campaign_end,
                                                 selected_regions, date_cols['ga'], region_cols['ga'], ga_split_col, use_pivot)
                    
                    def build_ratio_dict(num_dict, den_dict, num_metric_name, den_metric_name, ratio_name):
                        ratio_dict = {}
                        for (split_val, pivot_val, metric_name), num_val in num_dict.items():
                            if metric_name != num_metric_name:
                                continue
                            den_val = den_dict.get((split_val, pivot_val, den_metric_name), 0)
                            ratio_dict[(split_val, pivot_val, ratio_name)] = num_val / den_val if den_val != 0 else 0
                        return ratio_dict

                    # Calculate ratios by aligning group prefixes instead of full metric keys
                    ga_camp = build_ratio_dict(ga_camp_num, ga_camp_den, num_metric, den_metric, metric)
                    ga_pre = build_ratio_dict(ga_pre_num, ga_pre_den, num_metric, den_metric, metric)
                    
                    filtered_data['ga_campaign'] = None  # Ratio doesn't have raw data
                    filtered_data['ga_pre'] = None
                else:
                    continue
            else:
                if is_avg_session_duration:
                    ga_camp, filtered_data['ga_campaign'] = process_weighted_average(
                        st.session_state.data['ga'], metric, ga_sessions_col, campaign_start, campaign_end,
                        selected_regions, date_cols['ga'], region_cols['ga'], ga_split_col, use_pivot)
                    ga_pre, filtered_data['ga_pre'] = process_weighted_average(
                        st.session_state.data['ga'], metric, ga_sessions_col, pre_campaign_start, pre_campaign_end,
                        selected_regions, date_cols['ga'], region_cols['ga'], ga_split_col, use_pivot)
                else:
                    # Regular metric
                    ga_camp, filtered_data['ga_campaign'] = process_data(st.session_state.data['ga'], [metric], campaign_start, campaign_end,
                                          selected_regions, date_cols['ga'], region_cols['ga'], ga_split_col, use_pivot)
                    ga_pre, filtered_data['ga_pre'] = process_data(st.session_state.data['ga'], [metric], pre_campaign_start, pre_campaign_end,
                                         selected_regions, date_cols['ga'], region_cols['ga'], ga_split_col, use_pivot)
            
            ga_camp_index = build_index(ga_camp)
            ga_pre_index = build_index(ga_pre)

            if use_pivot:
                pivot_values = set()
                for (_, pivot_val, _) in list(ga_camp.keys()) + list(ga_pre.keys()):
                    if pivot_val is not None:
                        pivot_values.add(pivot_val)
                
                for piv_val in pivot_values:
                    if ga_split_col != 'None':
                        row = {
                            'Data source': 'GA',
                            'Metric': f"{piv_val}_{metric}",
                            'Meta (Pre-Campaign)': 0,
                            'Non-meta (Pre-Campaign)': 0,
                            'Meta (Campaign)': 0,
                            'Non-meta (Campaign)': 0
                        }
                        row['Meta (Campaign)'] = get_index_value(ga_camp_index, metric, piv_val, 'meta') / region_divisor
                        row['Non-meta (Campaign)'] = get_index_value(ga_camp_index, metric, piv_val, 'non-meta') / region_divisor
                        row['Meta (Pre-Campaign)'] = get_index_value(ga_pre_index, metric, piv_val, 'meta') / region_divisor
                        row['Non-meta (Pre-Campaign)'] = get_index_value(ga_pre_index, metric, piv_val, 'non-meta') / region_divisor
                    else:
                        row = {
                            'Data source': 'GA',
                            'Metric': f"{piv_val}_{metric}",
                            'Meta (Pre-Campaign)': get_index_value(ga_pre_index, metric, piv_val, None) / region_divisor,
                            'Non-meta (Pre-Campaign)': '',
                            'Meta (Campaign)': get_index_value(ga_camp_index, metric, piv_val, None) / region_divisor,
                            'Non-meta (Campaign)': ''
                        }
                    results.append(row)
            elif ga_split_col != 'None':
                row = {
                    'Data source': 'GA',
                    'Metric': metric,
                    'Meta (Pre-Campaign)': get_index_value(ga_pre_index, metric, None, 'meta') / region_divisor,
                    'Non-meta (Pre-Campaign)': get_index_value(ga_pre_index, metric, None, 'non-meta') / region_divisor,
                    'Meta (Campaign)': get_index_value(ga_camp_index, metric, None, 'meta') / region_divisor,
                    'Non-meta (Campaign)': get_index_value(ga_camp_index, metric, None, 'non-meta') / region_divisor
                }
            else:
                row = {
                    'Data source': 'GA',
                    'Metric': metric,
                    'Meta (Pre-Campaign)': get_index_value(ga_pre_index, metric, None, None) / region_divisor,
                    'Non-meta (Pre-Campaign)': '',
                    'Meta (Campaign)': get_index_value(ga_camp_index, metric, None, None) / region_divisor,
                    'Non-meta (Campaign)': ''
                }
            results.append(row)
        
        results_df = pd.DataFrame(results)
        
        # Remove duplicate rows
        results_df = results_df.drop_duplicates(subset=['Data source', 'Metric'], keep='first')
        
        st.header("📊 Results")
        
        period_label_campaign = f"Campaign {campaign_start.strftime('%b %d')}-{campaign_end.strftime('%b %d')}"
        period_label_pre = f"Pre-Campaign {pre_campaign_start.strftime('%b %d')}-{pre_campaign_end.strftime('%b %d')}"
        
        # Group by data source and display with beautiful HTML
        grouped = results_df.groupby('Data source')
        cols = st.columns(3)
        
        for idx, (source, group_df) in enumerate(grouped):
            with cols[idx]:
                html = f"""
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
                    .results-table-{idx} {{
                        width: 100%;
                        border-collapse: separate;
                        border-spacing: 0;
                        margin: 10px 0;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                        border-radius: 6px;
                        overflow: hidden;
                        font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
                        font-size: 14px;
                        line-height: 1.25;
                    }}
                    .results-table-{idx} thead {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }}
                    .results-table-{idx} th {{
                        padding: 12px 10px;
                        text-align: center;
                        font-weight: 600;
                        font-size: 13px;
                        border: none;
                    }}
                    .results-table-{idx} tbody tr {{
                        transition: all 0.2s;
                    }}
                    .results-table-{idx} tbody tr:hover {{
                        background-color: #f8f9fa;
                    }}
                    .results-table-{idx} td {{
                        padding: 10px 12px;
                        border-bottom: 1px solid #e9ecef;
                        font-size: 13px;
                    }}
                    .results-table-{idx} td:first-child {{
                        font-weight: 500;
                        color: #495057;
                        font-size: 12px;
                    }}
                    .results-table-{idx} td:not(:first-child) {{
                        text-align: right;
                        font-family: 'IBM Plex Mono', 'Consolas', monospace;
                        font-variant-numeric: tabular-nums;
                    }}
                    .ratio-row-{idx} {{
                        background-color: #e9ecef;
                        font-style: normal;
                    }}
                    .meta-col-{idx} {{
                        background-color: #e3f2fd;
                    }}
                    .non-meta-col-{idx} {{
                        background-color: #fff3e0;
                    }}
                    .index-note-{idx} {{
                        font-size: 12px;
                        color: #5f6b7a;
                        text-align: center;
                        margin-top: -2px;
                        margin-bottom: 8px;
                    }}
                    .index-tag-{idx} {{
                        font-size: 11px;
                        color: #5f6b7a;
                        margin-left: 4px;
                    }}
                    .campaign-header-{idx} {{
                        background-color: #FFD966 !important;
                        color: #333 !important;
                    }}
                    .pre-campaign-header-{idx} {{
                        background-color: #4472C4 !important;
                    }}
                    .source-title-{idx} {{
                        font-size: 17px;
                        font-weight: 600;
                        color: #667eea;
                        margin-bottom: 8px;
                        text-align: center;
                    }}
                </style>
                <div class="source-title-{idx}">{source}</div>
                <div class="index-note-{idx}">Index in parentheses = Base 100. Pre shows (100); Campaign shows Campaign vs Pre. Meta compares Meta, Non-meta compares Non-meta.</div>
                <table class="results-table-{idx}">
                    <thead>
                        <tr>
                            <th rowspan="2">Metric</th>
                            <th colspan="2" class="pre-campaign-header-{idx}">{period_label_pre}</th>
                            <th colspan="2" class="campaign-header-{idx}">{period_label_campaign}</th>
                        </tr>
                        <tr>
                            <th class="pre-campaign-header-{idx}">Meta</th>
                            <th class="pre-campaign-header-{idx}">Non-meta</th>
                            <th class="campaign-header-{idx}">Meta</th>
                            <th class="campaign-header-{idx}">Non-meta</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for _, row in group_df.iterrows():
                    metric = row['Metric']
                    is_ratio = any(x in str(metric).lower() for x in ['ratio', 'duration', '/', '%', 'per', 'average'])
                    row_class = f'ratio-row-{idx}' if is_ratio else ''
                    
                    html += f'<tr class="{row_class}">'
                    html += f'<td>{metric}</td>'
                    
                    def format_val(val):
                        if val == '' or pd.isna(val):
                            return ''
                        if abs(val) < 1 and abs(val) > 0:
                            return f'{val:.1%}'
                        elif abs(val) > 100:
                            return f'{val:,.0f}'
                        else:
                            return f'{val:.2f}'
                    
                    def format_with_index(val, base_val):
                        """Format value with index in parentheses (base = 100)"""
                        if val == '' or pd.isna(val) or base_val == '' or pd.isna(base_val) or base_val == 0:
                            return format_val(val)
                        
                        formatted = format_val(val)
                        index = int((val / base_val) * 100)
                        return f'{formatted} <span class="index-tag-{idx}">({index})</span>'

                    def format_base_index(val):
                        """Format pre-campaign values with base index 100"""
                        if val == '' or pd.isna(val):
                            return ''
                        formatted = format_val(val)
                        return f'{formatted} <span class="index-tag-{idx}">(100)</span>'
                    
                    # Pre-Campaign values (base = 100)
                    meta_pre = row["Meta (Pre-Campaign)"]
                    non_meta_pre = row["Non-meta (Pre-Campaign)"]
                    meta_camp = row["Meta (Campaign)"]
                    non_meta_camp = row["Non-meta (Campaign)"]
                    
                    html += f'<td class="meta-col-{idx}">{format_base_index(meta_pre)}</td>'
                    html += f'<td class="non-meta-col-{idx}">{format_base_index(non_meta_pre)}</td>'
                    html += f'<td class="meta-col-{idx}">{format_with_index(meta_camp, meta_pre)}</td>'
                    html += f'<td class="non-meta-col-{idx}">{format_with_index(non_meta_camp, non_meta_pre)}</td>'
                    html += '</tr>'
                
                html += '</tbody></table>'
                st.markdown(html, unsafe_allow_html=True)
        
        # Download
        st.markdown("---")
        csv = results_df.to_csv(index=False)
        st.download_button("📥 Download Results as CSV", csv, "marketing_report.csv", "text/csv", use_container_width=True)
        
        # Debug expander with filtered data
else:
    st.info("Please upload all three data sources to continue.")
