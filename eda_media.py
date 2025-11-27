"""
EDA Insights - Professional Data Analysis Platform

A comprehensive Streamlit application for data exploration, visualization,
and insight management with advanced analytical capabilities.

Author: Production Team
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import io
import traceback
from pathlib import Path
from scipy import stats
import hashlib

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EDA Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for professional styling"""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global styles */
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Color scheme variables */
        :root {
            --primary-yellow: #FFBD59;
            --yellow-light: #FFCF87;
            --yellow-lighter: #FFE7C2;
            --yellow-lightest: #FFF2DF;
            --secondary-green: #41C185;
            --secondary-blue: #458EE2;
            --secondary-purple: #9B59B6;
            --secondary-red: #E74C3C;
            --text-dark: #333333;
            --text-medium: #666666;
            --text-light: #999999;
            --background-white: #FFFFFF;
            --background-light: #F5F5F5;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }
        
        /* Main app background */
        .main {
            background-color: var(--background-light);
        }
        
        /* Card component styling */
        .card {
            background: var(--background-white);
            border-radius: 16px;
            padding: 28px;
            margin: 12px 0;
            box-shadow: var(--shadow-lg);
            border: 1px solid var(--yellow-lighter);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, var(--primary-yellow) 0%, var(--secondary-green) 100%);
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-xl);
            border-color: var(--primary-yellow);
        }
        
        .card-title {
            color: var(--text-dark);
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 18px;
            letter-spacing: -0.02em;
        }
        
        /* Compact card variant */
        .card-compact {
            background: var(--background-white);
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--yellow-lighter);
        }
        
        /* Header styling */
        .app-header {
            text-align: center;
            background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--yellow-light) 100%);
            color: var(--text-dark);
            padding: 32px 20px;
            margin: -20px 0 32px 0;
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
        }
        
        .app-header h1 {
            font-size: 2.5em;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
        }
        
        .app-header p {
            font-size: 1.1em;
            margin-top: 8px;
            color: var(--text-dark);
            font-weight: 500;
            opacity: 0.9;
        }
        
        /* Section header */
        .section-header {
            background: linear-gradient(135deg, var(--yellow-lightest) 0%, var(--background-white) 100%);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid var(--primary-yellow);
            margin-bottom: 24px;
        }
        
        .section-header h2 {
            color: var(--text-dark);
            margin: 0;
            font-size: 1.8em;
        }
        
        .section-header p {
            color: var(--text-medium);
            margin: 8px 0 0 0;
            font-size: 1em;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--yellow-light) 100%);
            color: var(--text-dark);
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 1em;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-md);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--yellow-light) 0%, var(--primary-yellow) 100%);
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }
        
        /* Primary button variant */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--secondary-green) 0%, #35a372 100%);
            color: white;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #35a372 0%, var(--secondary-green) 100%);
        }
        
        /* Secondary button variant */
        .stButton > button[kind="secondary"] {
            background: linear-gradient(135deg, var(--secondary-blue) 0%, #3a7bc8 100%);
            color: white;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #3a7bc8 0%, var(--secondary-blue) 100%);
        }
        
        /* Enhanced metric cards */
        .metric-card {
            background: linear-gradient(135deg, var(--background-white) 0%, var(--yellow-lightest) 100%);
            padding: 20px;
            border-radius: 12px;
            box-shadow: var(--shadow-md);
            border: 2px solid var(--yellow-lighter);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-xl);
            border-color: var(--primary-yellow);
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: 700;
            color: var(--text-dark);
            margin: 8px 0;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: var(--text-medium);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Info styling */
        .stInfo {
            background-color: var(--yellow-lightest);
            border-left-color: var(--secondary-blue);
        }
        
        /* Success styling */
        .stSuccess {
            background-color: #e8f5f0;
            border-left-color: var(--secondary-green);
        }
        
        /* Warning styling */
        .stWarning {
            background-color: #fff4e6;
            border-left-color: var(--primary-yellow);
        }
        
        /* Metric styling */
        .stMetric {
            background: var(--background-white);
            padding: 16px;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--yellow-lightest);
        }
        
        /* Divider styling */
        hr {
            border-color: var(--yellow-lighter);
            margin: 24px 0;
        }
        
        /* Dataframe styling */
        .dataframe {
            border: 1px solid var(--yellow-lighter) !important;
            border-radius: 8px;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: var(--background-white);
            padding: 8px;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            color: var(--text-medium);
            font-weight: 600;
            padding: 12px 24px;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--yellow-light) 100%);
            color: var(--text-dark);
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: var(--background-white);
            border-right: 2px solid var(--yellow-lighter);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--yellow-lightest);
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Select box styling */
        .stSelectbox > div > div {
            border-color: var(--yellow-lighter);
            border-radius: 8px;
        }
        
        /* Multiselect styling */
        .stMultiSelect > div > div {
            border-color: var(--yellow-lighter);
            border-radius: 8px;
        }
        
        /* Text input styling */
        .stTextInput > div > div {
            border-color: var(--yellow-lighter);
            border-radius: 8px;
        }
        
        /* Date input styling */
        .stDateInput > div > div {
            border-color: var(--yellow-lighter);
            border-radius: 8px;
        }
        
        /* Active dataset indicator */
        .active-dataset {
            background: linear-gradient(135deg, var(--secondary-green) 0%, #35a372 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            display: inline-block;
            margin-left: 8px;
        }
        
        /* Badge */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 0 4px;
        }
        
        .badge-success {
            background: var(--secondary-green);
            color: white;
        }
        
        .badge-info {
            background: var(--secondary-blue);
            color: white;
        }
        
        .badge-warning {
            background: var(--primary-yellow);
            color: var(--text-dark);
        }
        
        /* Configuration panel */
        .config-panel {
            background: var(--yellow-lightest);
            padding: 20px;
            border-radius: 12px;
            border: 2px solid var(--yellow-lighter);
            margin-bottom: 16px;
        }
        
        .config-section {
            margin-bottom: 16px;
        }
        
        .config-section-title {
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 8px;
            font-size: 1em;
        }
        </style>
    """, unsafe_allow_html=True)

# Inject custom CSS
inject_custom_css()

COMMENTS_FILE = "eda_comments.json"
ANALYSIS_TABS = ["overview", "explore", "pivot", "correlation", "clustering", "feature_importance"]
DEFAULT_SHEET_NAME = "Sheet 1"

# ============================================================================
# COMMENT MANAGEMENT SYSTEM
# ============================================================================

class CommentManager:
    """Comment management with file-based persistence"""
    
    @staticmethod
    def load_comments():
        """Load all comments with error handling"""
        try:
            if not Path(COMMENTS_FILE).exists():
                return pd.DataFrame(), None
            
            with open(COMMENTS_FILE, 'r', encoding='utf-8') as f:
                comments = json.load(f)
            
            if not comments:
                return pd.DataFrame(), None
                
            df = pd.DataFrame(comments)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df, None
        except Exception as e:
            return pd.DataFrame(), f"Error loading comments: {str(e)}"
    
    @staticmethod
    def save_comment(comment_text, tab_name, context_data=None):
        """Save new comment with validation"""
        try:
            if not comment_text or not comment_text.strip():
                return False, "Comment text cannot be empty"
            
            df, error = CommentManager.load_comments()
            if error:
                df = pd.DataFrame()
            
            new_comment = {
                'id': datetime.now().strftime('%Y%m%d%H%M%S%f'),
                'timestamp': datetime.now().isoformat(),
                'comment_text': comment_text.strip(),
                'tab_name': tab_name,
                'context_data': json.dumps(context_data) if context_data else None
            }
            
            if df.empty:
                df = pd.DataFrame([new_comment])
            else:
                df = pd.concat([df, pd.DataFrame([new_comment])], ignore_index=True)
            
            comments_list = df.to_dict('records')
            for c in comments_list:
                if isinstance(c['timestamp'], pd.Timestamp):
                    c['timestamp'] = c['timestamp'].isoformat()
            
            with open(COMMENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(comments_list, f, indent=2, ensure_ascii=False)
            
            return True, "Comment saved successfully"
        except Exception as e:
            return False, f"Failed to save comment: {str(e)}"
    
    @staticmethod
    def delete_comment(comment_id):
        """Delete comment by ID"""
        try:
            df, error = CommentManager.load_comments()
            if error:
                return False, error
            
            if df.empty:
                return False, "No comments found"
            
            df = df[df['id'] != comment_id]
            
            comments_list = df.to_dict('records')
            for c in comments_list:
                if isinstance(c['timestamp'], pd.Timestamp):
                    c['timestamp'] = c['timestamp'].isoformat()
            
            with open(COMMENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(comments_list, f, indent=2, ensure_ascii=False)
            
            return True, "Comment deleted successfully"
        except Exception as e:
            return False, f"Failed to delete comment: {str(e)}"
    
    @staticmethod
    def export_comments():
        """Export all comments as CSV"""
        try:
            df, error = CommentManager.load_comments()
            if error:
                return None, error
            
            if df.empty:
                return None, "No comments to export"
            
            return df.to_csv(index=False), None
        except Exception as e:
            return None, f"Export failed: {str(e)}"

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

class DataLoader:
    """Robust data loading with comprehensive error handling"""
    
    @staticmethod
    def parse_dates(df, date_column):
        """Attempt multiple date parsing strategies"""
        parse_strategies = [
            lambda col: pd.to_datetime(col, infer_datetime_format=True, errors='coerce'),
            lambda col: pd.to_datetime(col, format='%d-%m-%Y', errors='coerce'),
            lambda col: pd.to_datetime(col, format='%d/%m/%Y', errors='coerce'),
            lambda col: pd.to_datetime(col, format='%m/%d/%Y', errors='coerce'),
            lambda col: pd.to_datetime(col, format='%Y-%m-%d', errors='coerce'),
            lambda col: pd.to_datetime(col, format='%Y/%m/%d', errors='coerce'),
            lambda col: pd.to_datetime(col, dayfirst=True, errors='coerce'),
            lambda col: pd.to_datetime(col, format='mixed', errors='coerce'),
        ]
        
        for strategy in parse_strategies:
            try:
                parsed = strategy(df[date_column])
                
                if parsed.notna().sum() / len(parsed) > 0.5:
                    valid_dates = parsed[parsed.notna()]
                    if len(valid_dates) > 0:
                        year_1970_count = (valid_dates.dt.year == 1970).sum()
                        if year_1970_count / len(valid_dates) < 0.9:
                            df[date_column] = parsed
                            return True, None
            except:
                continue
        
        return False, f"Could not parse dates in column '{date_column}'"
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_file(_file, file_signature=None):
        """Load and validate data file"""
        try:
            _file.seek(0)
            
            if _file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(_file, encoding='utf-8')
                except UnicodeDecodeError:
                    _file.seek(0)
                    df = pd.read_csv(_file, encoding='latin-1')
            elif _file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(_file)
            else:
                return None, "Unsupported file format. Please upload CSV or Excel files."
            
            if df.empty:
                return None, "File is empty. Please upload a file with data."
            
            if len(df.columns) == 0:
                return None, "No columns found in file."
            
            # Date column detection
            exact_date_cols = [col for col in df.columns if col.lower() == 'date']
            date_in_name = [col for col in df.columns if 'date' in col.lower() and col.lower() != 'date']
            day_cols = [col for col in df.columns if col.lower() == 'day']
            other_date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['time', 'period', 'timestamp']) and col not in exact_date_cols + date_in_name + day_cols]
            
            date_candidates_all = exact_date_cols + date_in_name + day_cols + other_date_cols
            
            date_column_found = None
            invalid_date_columns = []
            
            if date_candidates_all:
                for date_col in date_candidates_all:
                    success, error = DataLoader.parse_dates(df, date_col)
                    
                    if success:
                        date_column_found = date_col
                        if date_col != 'Day':
                            df = df.rename(columns={date_col: 'Day'})
                        
                        df = df[df['Day'].notna()].copy()
                        df = df.sort_values('Day').reset_index(drop=True)
                        break
                    else:
                        invalid_date_columns.append(date_col)
            
            df.attrs['invalid_date_columns'] = invalid_date_columns
            if date_column_found:
                df.attrs['date_column_used'] = date_column_found
            
            return df, None
            
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg

# ============================================================================
# DATA PROCESSING UTILITIES
# ============================================================================

class DataProcessor:
    """Advanced data processing utilities"""
    
    @staticmethod
    def normalize_data(df, columns, method='minmax'):
        """Normalize data using various methods"""
        df_norm = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df_norm[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[f'{col}_norm'] = 0
                    
            elif method == 'zscore':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val != 0:
                    df_norm[f'{col}_norm'] = (df[col] - mean_val) / std_val
                else:
                    df_norm[f'{col}_norm'] = 0
                    
            elif method == 'pct_max':
                max_val = df[col].max()
                if max_val != 0:
                    df_norm[f'{col}_norm'] = (df[col] / max_val) * 100
                else:
                    df_norm[f'{col}_norm'] = 0
        
        return df_norm
    
    @staticmethod
    def calculate_moving_average(df, column, window=7):
        """Calculate moving average"""
        return df[column].rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def calculate_trend(df, column):
        """Calculate linear trend"""
        if len(df) < 2:
            return None
        
        x = np.arange(len(df))
        y = df[column].values
        
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return None
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'trend_line': slope * x + intercept
        }
    
    @staticmethod
    def create_pivot_with_totals(df, index, columns, values, aggfunc, show_percentages=False, pct_type='column'):
        """Create Excel-like pivot table with grand totals and percentages"""
        if columns:
            pivot = pd.pivot_table(
                df,
                values=values,
                index=index,
                columns=columns,
                aggfunc=aggfunc,
                fill_value=0,
                margins=True,
                margins_name='Grand Total'
            )
        else:
            pivot = df.groupby(index).agg({v: aggfunc for v in values})
            grand_total = df.agg({v: aggfunc for v in values})
            grand_total.name = 'Grand Total'
            pivot = pd.concat([pivot, pd.DataFrame([grand_total])])
        
        if show_percentages and len(pivot) > 1:
            if pct_type == 'column':
                pct_pivot = pivot.copy()
                for col in pct_pivot.columns:
                    if col != 'Grand Total' and 'Grand Total' in pct_pivot.index:
                        col_total = pct_pivot.loc['Grand Total', col]
                        if col_total != 0:
                            pct_pivot[col] = (pct_pivot[col] / col_total * 100).round(2)
                return pivot, pct_pivot
                
            elif pct_type == 'row':
                pct_pivot = pivot.copy()
                if 'Grand Total' in pct_pivot.columns:
                    row_totals = pct_pivot['Grand Total']
                    for idx in pct_pivot.index:
                        if row_totals[idx] != 0:
                            pct_pivot.loc[idx] = (pct_pivot.loc[idx] / row_totals[idx] * 100).round(2)
                return pivot, pct_pivot
                
            elif pct_type == 'grand':
                pct_pivot = pivot.copy()
                if 'Grand Total' in pct_pivot.index and 'Grand Total' in pct_pivot.columns:
                    grand_total = pct_pivot.loc['Grand Total', 'Grand Total']
                    if grand_total != 0:
                        pct_pivot = (pct_pivot / grand_total * 100).round(2)
                return pivot, pct_pivot
        
        return pivot, None

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.datasets = {}
        st.session_state.active_dataset_key = None
        st.session_state.editing_comment_id = None
        st.session_state.delete_confirm_id = None

initialize_session_state()

class DatasetManager:
    """Manages multiple datasets with isolated state"""
    
    @staticmethod
    def _create_default_worksheet_state():
        """Template for worksheet-specific state"""
        return {
            'order': [DEFAULT_SHEET_NAME],
            'active': DEFAULT_SHEET_NAME,
            'dataset_key': None,
            'sheet_filters': {
                DEFAULT_SHEET_NAME: {
                    'start_date': None,
                    'end_date': None,
                    'identifier_filters': {}
                }
            }
        }
    
    @staticmethod
    def _create_default_dataset_state():
        """Template for dataset-level storage"""
        return {
            'name': '',
            'df': None,
            'file_signature': None,
            'has_dates': False,
            'min_date': None,
            'max_date': None,
            'all_columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'excluded_cols': [],
            'columns_classified': False,
            'selected_identifiers': [],
            'selected_metrics': [],
            'worksheets': {
                tab: DatasetManager._create_default_worksheet_state()
                for tab in ANALYSIS_TABS
            }
        }
    
    @staticmethod
    def reset_dataset(dataset_key):
        """Reset dataset storage when source file changes"""
        st.session_state.datasets[dataset_key] = DatasetManager._create_default_dataset_state()
    
    @staticmethod
    def clear_column_selection_state(dataset_key):
        """Remove identifier/metric widget selections"""
        for prefix in ['identifiers', 'metrics']:
            state_key = f"{prefix}_{dataset_key}"
            if state_key in st.session_state:
                del st.session_state[state_key]
    
    @staticmethod
    def create_dataset_key(file_name, file_size, index):
        """Create unique dataset key"""
        return f"{file_name}_{file_size}_{index}"
    
    @staticmethod
    def get_dataset(dataset_key):
        """Get dataset state"""
        if dataset_key not in st.session_state.datasets:
            st.session_state.datasets[dataset_key] = DatasetManager._create_default_dataset_state()
        dataset = st.session_state.datasets[dataset_key]
        if 'file_signature' not in dataset:
            dataset['file_signature'] = None
        return dataset
    
    @staticmethod
    def set_active_dataset(dataset_key):
        """Switch active dataset"""
        if dataset_key in st.session_state.datasets:
            st.session_state.active_dataset_key = dataset_key
            return True
        return False
    
    @staticmethod
    def get_active_dataset():
        """Get currently active dataset"""
        if st.session_state.active_dataset_key:
            return DatasetManager.get_dataset(st.session_state.active_dataset_key)
        return None
    
    @staticmethod
    def update_dataset(dataset_key, **kwargs):
        """Update dataset properties"""
        ds = DatasetManager.get_dataset(dataset_key)
        for key, value in kwargs.items():
            ds[key] = value
    
    @staticmethod
    def get_worksheet_state(dataset_key, tab_key):
        """Get worksheet state for specific dataset and tab"""
        ds = DatasetManager.get_dataset(dataset_key)
        if tab_key not in ds['worksheets']:
            ds['worksheets'][tab_key] = {
                'order': [DEFAULT_SHEET_NAME],
                'active': DEFAULT_SHEET_NAME,
                'dataset_key': None,
                'sheet_filters': {
                    DEFAULT_SHEET_NAME: {
                        'start_date': None,
                        'end_date': None,
                        'identifier_filters': {}
                    }
                }
            }
        return ds['worksheets'][tab_key]
    
    @staticmethod
    def add_worksheet(dataset_key, tab_key, sheet_name):
        """Add new worksheet"""
        ws_state = DatasetManager.get_worksheet_state(dataset_key, tab_key)
        if sheet_name not in ws_state['order']:
            ws_state['order'].append(sheet_name)
            ws_state['sheet_filters'][sheet_name] = {
                'start_date': None,
                'end_date': None,
                'identifier_filters': {}
            }
    
    @staticmethod
    def set_active_worksheet(dataset_key, tab_key, sheet_name):
        """Set active worksheet"""
        ws_state = DatasetManager.get_worksheet_state(dataset_key, tab_key)
        if sheet_name in ws_state['order']:
            ws_state['active'] = sheet_name
    
    @staticmethod
    def get_sheet_filters(dataset_key, tab_key, sheet_name):
        """Get filters for specific sheet"""
        ws_state = DatasetManager.get_worksheet_state(dataset_key, tab_key)
        if sheet_name not in ws_state['sheet_filters']:
            ws_state['sheet_filters'][sheet_name] = {
                'start_date': None,
                'end_date': None,
                'identifier_filters': {}
            }
        return ws_state['sheet_filters'][sheet_name]
    
    @staticmethod
    def update_sheet_filters(dataset_key, tab_key, sheet_name, **filters):
        """Update filters for specific sheet"""
        sheet_filters = DatasetManager.get_sheet_filters(dataset_key, tab_key, sheet_name)
        for key, value in filters.items():
            sheet_filters[key] = value
    
    @staticmethod
    def get_sheet_results(dataset_key, tab_key, sheet_name):
        """Get stored results for specific sheet"""
        ws_state = DatasetManager.get_worksheet_state(dataset_key, tab_key)
        if 'sheet_results' not in ws_state:
            ws_state['sheet_results'] = {}
        if sheet_name not in ws_state['sheet_results']:
            ws_state['sheet_results'][sheet_name] = {
                'pivot_generated': False,
                'pivot_config': {},
                'pivot_result': None,
                'pct_pivot': None,
                'clustering_run': False,
                'cluster_data': None
            }
        return ws_state['sheet_results'][sheet_name]
    
    @staticmethod
    def update_sheet_results(dataset_key, tab_key, sheet_name, **results):
        """Update stored results for specific sheet"""
        sheet_results = DatasetManager.get_sheet_results(dataset_key, tab_key, sheet_name)
        for key, value in results.items():
            sheet_results[key] = value
    
    @staticmethod
    def set_sheet_dataset(dataset_key, tab_key, sheet_name, sheet_dataset_key):
        """Set which dataset a sheet is viewing"""
        ws_state = DatasetManager.get_worksheet_state(dataset_key, tab_key)
        ws_state['dataset_key'] = sheet_dataset_key
    
    @staticmethod
    def get_sheet_dataset(dataset_key, tab_key, sheet_name):
        """Get the dataset key for a specific sheet"""
        ws_state = DatasetManager.get_worksheet_state(dataset_key, tab_key)
        # If no dataset set for sheet, use the current active dataset
        if ws_state.get('dataset_key') is None:
            return dataset_key
        return ws_state['dataset_key']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_value(value, col_name='', as_percentage=False):
    """Smart value formatting"""
    if pd.isna(value):
        return "N/A"
    
    try:
        value = float(value)
        
        if as_percentage:
            return f"{value:.1f}%"
        
        if any(x in col_name.lower() for x in ['amount', 'spent', 'price', 'revenue', 'sales', 'cost', 'usd', '$', 'dollar', 'euro', '€']):
            return f"${value:,.2f}"
        elif any(x in col_name.lower() for x in ['rate', 'percent', '%', 'ratio']):
            if value <= 1:
                return f"{value:.2%}"
            else:
                return f"{value:.2f}%"
        elif any(x in col_name.lower() for x in ['count', 'quantity', 'qty', 'number', 'total', 'impressions', 'clicks', 'views']):
            return f"{int(value):,}"
        elif abs(value) >= 1000 or value % 1 == 0:
            return f"{value:,.0f}"
        else:
            return f"{value:,.2f}"
    except:
        return str(value)

def aggregate_data(df, level, method='sum'):
    """Aggregate data by time period"""
    if 'Day' not in df.columns:
        return df
    
    df_agg = df.copy()
    
    if level == "Weekly":
        df_agg['Period'] = df_agg['Day'].dt.to_period('W').apply(lambda r: r.start_time)
    elif level == "Monthly":
        df_agg['Period'] = df_agg['Day'].dt.to_period('M').apply(lambda r: r.start_time)
    elif level == "Quarterly":
        df_agg['Period'] = df_agg['Day'].dt.to_period('Q').apply(lambda r: r.start_time)
    else:
        df_agg['Period'] = df_agg['Day']
    
    numeric_cols = df_agg.select_dtypes(include=[np.number]).columns
    agg_dict = {col: method for col in numeric_cols}
    
    return df_agg.groupby('Period').agg(agg_dict).reset_index()

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_dataset_selector(location="main"):
    """Render dataset selector"""
    if not st.session_state.datasets:
        st.info("📤 Upload files using the sidebar to get started")
        return None
    
    dataset_keys = list(st.session_state.datasets.keys())
    
    if not st.session_state.active_dataset_key or st.session_state.active_dataset_key not in dataset_keys:
        st.session_state.active_dataset_key = dataset_keys[0]
    
    current_idx = dataset_keys.index(st.session_state.active_dataset_key) if st.session_state.active_dataset_key in dataset_keys else 0
    
    selected_key = st.selectbox(
        "📁 Select Active Dataset",
        dataset_keys,
        index=current_idx,
        format_func=lambda k: f"{st.session_state.datasets[k]['name']} {'✅' if st.session_state.datasets[k]['columns_classified'] else '⚙️'}",
        key=f"dataset_selector_{location}"
    )
    
    if selected_key != st.session_state.active_dataset_key:
        DatasetManager.set_active_dataset(selected_key)
        st.rerun()
    
    return selected_key

def render_worksheet_selector(dataset_key, tab_key, show_dataset_selector=True):
    """Render worksheet selector with optional dataset selection per sheet"""
    ws_state = DatasetManager.get_worksheet_state(dataset_key, tab_key)
    
    col1, col2 = st.columns([5, 1])
    
    with col2:
        if st.button("➕", key=f"add_sheet_{dataset_key}_{tab_key}", help="Add new worksheet"):
            counter = len(ws_state['order']) + 1
            new_name = f"Sheet {counter}"
            while new_name in ws_state['order']:
                counter += 1
                new_name = f"Sheet {counter}"
            
            DatasetManager.add_worksheet(dataset_key, tab_key, new_name)
            DatasetManager.set_active_worksheet(dataset_key, tab_key, new_name)
            st.rerun()
    
    with col1:
        current_idx = ws_state['order'].index(ws_state['active']) if ws_state['active'] in ws_state['order'] else 0
        
        selected_sheet = st.radio(
            "Worksheets",
            ws_state['order'],
            index=current_idx,
            key=f"sheet_selector_{dataset_key}_{tab_key}",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if selected_sheet != ws_state['active']:
            DatasetManager.set_active_worksheet(dataset_key, tab_key, selected_sheet)
            st.rerun()
    
    # Dataset selector for this specific sheet
    if show_dataset_selector and len(st.session_state.datasets) > 1:
        st.markdown("---")
        dataset_keys = list(st.session_state.datasets.keys())
        current_dataset_key = DatasetManager.get_sheet_dataset(dataset_key, tab_key, selected_sheet)
        
        if current_dataset_key not in dataset_keys:
            current_dataset_key = dataset_key
        
        current_dataset_idx = dataset_keys.index(current_dataset_key)
        
        sheet_dataset = st.selectbox(
            f"📊 Dataset for {selected_sheet}",
            dataset_keys,
            index=current_dataset_idx,
            format_func=lambda k: st.session_state.datasets[k]['name'],
            key=f"sheet_dataset_{dataset_key}_{tab_key}_{selected_sheet}"
        )
        
        if sheet_dataset != current_dataset_key:
            DatasetManager.set_sheet_dataset(dataset_key, tab_key, selected_sheet, sheet_dataset)
            st.rerun()
        
        return selected_sheet, sheet_dataset
    
    return selected_sheet, dataset_key

def render_filter_controls(dataset_key, tab_key, sheet_name):
    """Render filter controls and return filtered dataframe"""
    ds = DatasetManager.get_dataset(dataset_key)
    df = ds['df']
    
    if df is None or df.empty:
        return pd.DataFrame(), {}, None, None
    
    sheet_filters = DatasetManager.get_sheet_filters(dataset_key, tab_key, sheet_name)
    
    if sheet_filters['start_date'] is None and ds['has_dates']:
        sheet_filters['start_date'] = ds['min_date']
        sheet_filters['end_date'] = ds['max_date']
    
    filtered_df = df.copy()
    active_filters = {}
    
    with st.expander("🎯 Data Filters", expanded=False):
        st.caption(f"Filters for: **{sheet_name}** (Dataset: {ds['name']})")
        
        if ds['has_dates'] and ds['min_date'] and ds['max_date']:
            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
            
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=sheet_filters['start_date'] or ds['min_date'],
                    min_value=ds['min_date'],
                    max_value=ds['max_date'],
                    key=f"start_{dataset_key}_{tab_key}_{sheet_name}"
                )
                sheet_filters['start_date'] = start_date
            
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=sheet_filters['end_date'] or ds['max_date'],
                    min_value=ds['min_date'],
                    max_value=ds['max_date'],
                    key=f"end_{dataset_key}_{tab_key}_{sheet_name}"
                )
                sheet_filters['end_date'] = end_date
            
            with col3:
                if st.button("30d", key=f"30d_{dataset_key}_{tab_key}_{sheet_name}"):
                    sheet_filters['start_date'] = max(ds['min_date'], ds['max_date'] - timedelta(days=30))
                    sheet_filters['end_date'] = ds['max_date']
                    st.rerun()
            
            with col4:
                if st.button("90d", key=f"90d_{dataset_key}_{tab_key}_{sheet_name}"):
                    sheet_filters['start_date'] = max(ds['min_date'], ds['max_date'] - timedelta(days=90))
                    sheet_filters['end_date'] = ds['max_date']
                    st.rerun()
            
            with col5:
                if st.button("All", key=f"all_{dataset_key}_{tab_key}_{sheet_name}"):
                    sheet_filters['start_date'] = ds['min_date']
                    sheet_filters['end_date'] = ds['max_date']
                    st.rerun()
            
            if start_date and end_date and start_date <= end_date:
                mask = (filtered_df['Day'].dt.date >= start_date) & (filtered_df['Day'].dt.date <= end_date)
                filtered_df = filtered_df[mask].copy()
        
        if ds['selected_identifiers']:
            st.divider()
            st.markdown("**Dimension Filters**")
            
            filter_cols = st.columns(min(3, len(ds['selected_identifiers'])))
            
            for idx, identifier in enumerate(ds['selected_identifiers'][:9]):
                with filter_cols[idx % len(filter_cols)]:
                    unique_vals = sorted(filtered_df[identifier].dropna().unique())
                    
                    if 0 < len(unique_vals) <= 100:
                        current_filter = sheet_filters['identifier_filters'].get(identifier, [])
                        
                        selected_vals = st.multiselect(
                            f"**{identifier}**",
                            options=unique_vals,
                            default=[v for v in current_filter if v in unique_vals],
                            key=f"filter_{dataset_key}_{tab_key}_{sheet_name}_{identifier}"
                        )
                        
                        if selected_vals:
                            sheet_filters['identifier_filters'][identifier] = selected_vals
                            active_filters[identifier] = selected_vals
                        elif identifier in sheet_filters['identifier_filters']:
                            del sheet_filters['identifier_filters'][identifier]
        
        for col, values in active_filters.items():
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    return filtered_df, active_filters, sheet_filters['start_date'], sheet_filters['end_date']

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.markdown("""
<div class="app-header">
    <h1>📊 EDA Insights</h1>
    <p>Professional Data Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR: FILE UPLOAD
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--yellow-light) 100%); 
         padding: 20px; border-radius: 12px; margin-bottom: 24px; text-align: center;">
        <h2 style="color: var(--text-dark); margin: 0; font-size: 1.5em;">📤 Data Upload</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload multiple files for parallel analysis"
    )
    
    if uploaded_files:
        current_keys = []
        
        for idx, file in enumerate(uploaded_files):
            dataset_key = DatasetManager.create_dataset_key(file.name, file.size, idx)
            current_keys.append(dataset_key)
            
            file_bytes = file.getvalue()
            file_signature = hashlib.md5(file_bytes).hexdigest()
            
            ds = DatasetManager.get_dataset(dataset_key)
            current_signature = ds.get('file_signature')
            
            if ds['df'] is not None and current_signature is None:
                DatasetManager.reset_dataset(dataset_key)
                DatasetManager.clear_column_selection_state(dataset_key)
                ds = DatasetManager.get_dataset(dataset_key)
                current_signature = ds.get('file_signature')
            
            if current_signature and current_signature != file_signature:
                DatasetManager.reset_dataset(dataset_key)
                DatasetManager.clear_column_selection_state(dataset_key)
                ds = DatasetManager.get_dataset(dataset_key)
            
            if ds['df'] is None:
                with st.spinner(f"Loading {file.name}..."):
                    buffer = io.BytesIO(file_bytes)
                    buffer.name = file.name
                    df, error = DataLoader.load_file(buffer, file_signature)
                    
                    if error:
                        st.error(f"❌ {file.name}: {error}")
                        continue
                    
                    DatasetManager.update_dataset(
                        dataset_key,
                        name=file.name,
                        df=df,
                        file_signature=file_signature
                    )
                    
                    has_dates = 'Day' in df.columns
                    min_date = None
                    max_date = None
                    
                    if has_dates:
                        min_date = df['Day'].min().date()
                        max_date = df['Day'].max().date()
                    
                    excluded_cols = ['Day']
                    if hasattr(df, 'attrs') and 'invalid_date_columns' in df.attrs:
                        excluded_cols.extend(df.attrs['invalid_date_columns'])
                    
                    all_columns = [col for col in df.columns if col not in excluded_cols]
                    
                    text_columns = [
                        col for col in df.select_dtypes(include=['object', 'category', 'string']).columns
                        if col not in excluded_cols
                    ]
                    bool_columns = [
                        col for col in df.select_dtypes(include=['bool']).columns
                        if col not in excluded_cols
                    ]
                    numeric_candidates = [
                        col for col in df.select_dtypes(include=[np.number]).columns
                        if col not in excluded_cols
                    ]
                    
                    identifier_keywords = [
                        'code', 'segment', 'group', 'category', 'region', 'country', 'city',
                        'area', 'team', 'channel', 'campaign', 'type', 'class', 'cluster',
                        'bucket', 'brand', 'product', 'store', 'market'
                    ]
                    metric_keywords = [
                        'amount', 'total', 'sum', 'conversion', 'rate', 'ctr', 'cpc', 'cpm',
                        'revenue', 'sale', 'profit', 'cost', 'spend', 'value', 'score',
                        'avg', 'average', 'mean', 'impression', 'click', 'view', 'time',
                        'duration', 'qty', 'quantity', 'volume', 'weight', 'height', 'age'
                    ]
                    
                    numeric_identifier_candidates = []
                    total_rows = len(df)
                    
                    for col in numeric_candidates:
                        col_lower = col.lower()
                        unique_vals = df[col].nunique(dropna=True)
                        unique_ratio = (unique_vals / total_rows) if total_rows else 0
                        
                        if any(metric_kw in col_lower for metric_kw in metric_keywords):
                            continue
                        
                        id_like_name = (
                            col_lower.endswith('id') or '_id' in col_lower or ' id' in col_lower
                            or any(keyword in col_lower for keyword in identifier_keywords)
                        )
                        
                        if unique_vals <= 15 or (unique_vals <= 50 and unique_ratio <= 0.1) or id_like_name:
                            numeric_identifier_candidates.append(col)
                    
                    categorical_candidates = []
                    for col in text_columns + bool_columns + numeric_identifier_candidates:
                        if col not in categorical_candidates:
                            categorical_candidates.append(col)
                    
                    numeric_columns = [col for col in numeric_candidates if col not in numeric_identifier_candidates]
                    categorical_columns = categorical_candidates
                    
                    DatasetManager.update_dataset(
                        dataset_key,
                        has_dates=has_dates,
                        min_date=min_date,
                        max_date=max_date,
                        all_columns=all_columns,
                        numeric_columns=numeric_columns,
                        categorical_columns=categorical_columns,
                        excluded_cols=excluded_cols
                    )
            
            size_kb = file.size / 1024
            status_icon = "✅" if ds['columns_classified'] else "⚙️"
            status_text = "Ready" if ds['columns_classified'] else "Setup Required"
            
            is_active = dataset_key == st.session_state.active_dataset_key
            active_badge = '<span class="active-dataset">ACTIVE</span>' if is_active else ''
            
            st.markdown(f"""
            <div style="background: {'var(--yellow-lightest)' if is_active else 'var(--background-light)'}; 
                 padding: 12px; border-radius: 8px; margin-bottom: 8px; 
                 border: 2px solid {'var(--secondary-green)' if is_active else 'transparent'};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>{status_icon} <strong>{file.name}</strong> {active_badge}</span>
                </div>
                <div style="color: var(--text-medium); font-size: 0.85em; margin-top: 4px;">
                    {size_kb:.1f} KB • {status_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        keys_to_remove = [k for k in st.session_state.datasets.keys() if k not in current_keys]
        for key in keys_to_remove:
            DatasetManager.clear_column_selection_state(key)
            del st.session_state.datasets[key]
            if st.session_state.active_dataset_key == key:
                st.session_state.active_dataset_key = None
        
        if not st.session_state.active_dataset_key and current_keys:
            st.session_state.active_dataset_key = current_keys[0]
    
    st.divider()
    
    st.markdown("### 📥 Sample Data")
    sample_df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=90),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 90),
        'Product': np.random.choice(['Product A', 'Product B', 'Product C'], 90),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Food'], 90),
        'Sales': np.random.randint(1000, 10000, 90),
        'Quantity': np.random.randint(10, 200, 90),
        'Revenue': np.random.uniform(5000, 50000, 90).round(2),
        'Cost': np.random.uniform(2000, 30000, 90).round(2)
    })
    
    st.download_button(
        '📥 Download Sample',
        sample_df.to_csv(index=False),
        "sample_data.csv",
        "text/csv",
        use_container_width=True
    )
    
    if st.session_state.datasets:
        st.divider()
        if st.button("🔄 Reset All", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ============================================================================
# CHECK IF FILES UPLOADED
# ============================================================================

if not st.session_state.datasets:
    st.markdown("""
    <div class="card" style="max-width: 800px; margin: 60px auto;">
        <div class="card-title" style="text-align: center; font-size: 2em;">
            🚀 Welcome to EDA Insights
        </div>
        <p style="text-align: center; color: var(--text-medium); font-size: 1.1em; margin: 20px 0;">
            Upload CSV or Excel files using the sidebar to begin your data analysis
        </p>
        <div style="text-align: center; margin-top: 30px;">
            <span class="badge badge-success">✨ Multi-file Support</span>
            <span class="badge badge-info">📊 Advanced Analytics</span>
            <span class="badge badge-warning">📈 Interactive Visualizations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================================
# DATASET SELECTION & CLASSIFICATION
# ============================================================================

st.markdown("---")

selected_key = render_dataset_selector("main")
if not selected_key:
    st.stop()

ds = DatasetManager.get_active_dataset()
if not ds:
    st.error("❌ No active dataset")
    st.stop()

df = ds['df']
if df is None or df.empty:
    st.error("❌ Dataset not loaded properly")
    st.stop()

# Dataset info
info_cols = st.columns(6)
with info_cols[0]:
    st.metric("📄 Rows", f"{len(df):,}")
with info_cols[1]:
    st.metric("📋 Columns", len(df.columns))
with info_cols[2]:
    st.metric("🔢 Numeric", len(ds['numeric_columns']))
with info_cols[3]:
    st.metric("🏷️ Categorical", len(ds['categorical_columns']))
with info_cols[4]:
    if ds['has_dates']:
        date_range = (ds['max_date'] - ds['min_date']).days
        st.metric("📅 Date Span", f"{date_range} days")
    else:
        st.metric("📅 Dates", "Not found")
with info_cols[5]:
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
    st.metric("💾 Memory", f"{memory_usage:.1f} MB")

if ds['has_dates'] and ds['min_date'] and ds['max_date']:
    st.info(f"📅 **Date Range:** {ds['min_date']} to {ds['max_date']}")

st.markdown("---")

# Column Classification
st.markdown("""
<div class="section-header">
    <h2>🎯 Column Classification</h2>
    <p>Define identifiers (dimensions) and metrics for <strong>{}</strong></p>
</div>
""".format(ds['name']), unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏷️ Identifiers")
    st.caption("Categorical columns for grouping and segmentation")
    
    available_identifiers = [c for c in ds['all_columns'] if c not in ds['selected_metrics']]
    
    selected_identifiers = st.multiselect(
        "Select identifier columns",
        options=available_identifiers,
        default=[c for c in ds['selected_identifiers'] if c in available_identifiers],
        key=f"identifiers_{selected_key}"
    )

with col2:
    st.subheader("📊 Metrics")
    st.caption("Numeric columns for quantitative analysis")
    
    available_metrics = [c for c in ds['all_columns'] if c not in selected_identifiers]
    
    selected_metrics = st.multiselect(
        "Select metric columns",
        options=available_metrics,
        default=[c for c in ds['selected_metrics'] if c in available_metrics],
        key=f"metrics_{selected_key}"
    )

action_col1, action_col2, action_col3 = st.columns([1, 1, 3])

with action_col1:
    if st.button("✅ Apply Configuration", type="primary", use_container_width=True):
        if not selected_metrics:
            st.error("❌ Select at least one metric")
        else:
            DatasetManager.update_dataset(
                selected_key,
                columns_classified=True,
                selected_identifiers=selected_identifiers,
                selected_metrics=selected_metrics
            )
            st.success(f"✅ Configuration saved")
            st.rerun()

with action_col2:
    if st.button("🔄 Auto-Configure", use_container_width=True):
        DatasetManager.update_dataset(
            selected_key,
            columns_classified=True,
            selected_identifiers=ds['categorical_columns'],
            selected_metrics=ds['numeric_columns'][:min(10, len(ds['numeric_columns']))]
        )
        st.success(f"✅ Auto-configured")
        st.rerun()

if ds['columns_classified']:
    st.success(f"✅ **{len(ds['selected_identifiers'])} Identifiers, {len(ds['selected_metrics'])} Metrics** configured")

st.markdown("---")

if not ds['columns_classified'] or not ds['selected_metrics']:
    st.warning("⚠️ Please configure columns and click 'Apply Configuration' to continue")
    st.stop()

# ============================================================================
# ANALYSIS TABS
# ============================================================================

tabs = st.tabs([
    "📊 Overview",
    "📈 Trend Analysis",
    "🔄 Pivot Analysis",
    "🔗 Correlation",
    "🎯 Clustering",
    "📋 Insights"
])

# TAB 1: OVERVIEW
with tabs[0]:
    st.markdown("""
    <div class="section-header">
        <h2>📊 Data Overview</h2>
        <p>Summary statistics and data preview</p>
    </div>
    """, unsafe_allow_html=True)
    
    result = render_worksheet_selector(selected_key, "overview", show_dataset_selector=True)
    if isinstance(result, tuple):
        active_sheet, sheet_dataset_key = result
    else:
        active_sheet = result
        sheet_dataset_key = selected_key
    
    # Use the sheet's selected dataset
    sheet_ds = DatasetManager.get_dataset(sheet_dataset_key)
    filtered_df, filters, start, end = render_filter_controls(sheet_dataset_key, "overview", active_sheet)
    
    if filtered_df.empty:
        st.warning("📊 No data matches the current filters")
        st.stop()
    
    # Metrics Summary
    st.markdown('<div class="card"><div class="card-title">📈 Key Metrics</div>', unsafe_allow_html=True)
    metric_cols = st.columns(min(len(sheet_ds['selected_metrics']), 6))
    for idx, metric in enumerate(sheet_ds['selected_metrics'][:6]):
        with metric_cols[idx]:
            total = filtered_df[metric].sum()
            avg = filtered_df[metric].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{metric}</div>
                <div class="metric-value">{format_value(total, metric)}</div>
                <div style="color: var(--text-light); font-size: 0.85em;">Avg: {format_value(avg, metric)}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistical Summary
    st.markdown('<div class="card"><div class="card-title">📊 Statistical Summary</div>', unsafe_allow_html=True)
    summary_df = filtered_df[sheet_ds['selected_metrics']].describe().T
    summary_df['median'] = filtered_df[sheet_ds['selected_metrics']].median()
    summary_df = summary_df[['count', 'mean', 'median', 'std', 'min', 'max']]
    
    for col in summary_df.columns:
        if col == 'count':
            summary_df[col] = summary_df[col].apply(lambda x: f"{int(x):,}")
        else:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:,.2f}")
    
    st.dataframe(summary_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Data preview
    st.markdown('<div class="card"><div class="card-title">📋 Data Preview</div>', unsafe_allow_html=True)
    display_cols = (['Day'] if sheet_ds['has_dates'] else []) + sheet_ds['selected_identifiers'] + sheet_ds['selected_metrics']
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[display_cols], use_container_width=True, height=400)
    
    csv = filtered_df[display_cols].to_csv(index=False)
    st.download_button("📥 Download Data", csv, f"data_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: TREND ANALYSIS
with tabs[1]:
    st.markdown("""
    <div class="section-header">
        <h2>📈 Trend Analysis</h2>
        <p>Time-series visualization with normalization and trend analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    result = render_worksheet_selector(selected_key, "explore", show_dataset_selector=True)
    if isinstance(result, tuple):
        active_sheet, sheet_dataset_key = result
    else:
        active_sheet = result
        sheet_dataset_key = selected_key
    
    sheet_ds = DatasetManager.get_dataset(sheet_dataset_key)
    
    if not sheet_ds['has_dates']:
        st.warning("⚠️ Time-series analysis requires a date column")
        st.stop()
    
    filtered_df, filters, start, end = render_filter_controls(sheet_dataset_key, "explore", active_sheet)
    
    if filtered_df.empty:
        st.warning("📊 No data matches the current filters")
        st.stop()
    
    # SIDE-BY-SIDE LAYOUT: Configuration (Left) | Visualization (Right)
    config_col, viz_col = st.columns([1, 2.5])
    
    with config_col:
        st.markdown("### ⚙️ Visualization Settings")
        
        # Time & Aggregation
        st.markdown("**📅 Time Settings**")
        agg_level = st.selectbox(
            "Time Period",
            ["Daily", "Weekly", "Monthly", "Quarterly"],
            key=f"explore_agg_{sheet_dataset_key}_{active_sheet}"
        )
        
        agg_method = st.selectbox(
            "Aggregation Method",
            ["Sum", "Mean", "Median", "Min", "Max"],
            key=f"explore_method_{sheet_dataset_key}_{active_sheet}"
        )
        
        st.divider()
        
        # Metrics Selection
        st.markdown("**📊 Metrics**")
        chart_metrics = st.multiselect(
            "Select Metrics",
            sheet_ds['selected_metrics'],
            default=sheet_ds['selected_metrics'][:min(3, len(sheet_ds['selected_metrics']))],
            key=f"explore_metrics_{sheet_dataset_key}_{active_sheet}",
            help="Choose metrics to visualize"
        )
        
        st.divider()
        
        # Chart Options
        st.markdown("**📈 Chart Options**")
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Bar", "Area", "Scatter"],
            key=f"explore_chart_{sheet_dataset_key}_{active_sheet}"
        )
        
        show_trend = st.checkbox(
            "Show Trend Line",
            value=False,
            key=f"explore_trend_{sheet_dataset_key}_{active_sheet}",
            help="Add linear regression trend"
        )
        
        st.divider()
        
        # Normalization Options
        st.markdown("**✨ Normalization**")
        enable_normalization = st.checkbox(
            "Enable Normalization",
            value=False,
            key=f"explore_norm_{sheet_dataset_key}_{active_sheet}",
            help="Scale metrics for comparison"
        )
        
        if enable_normalization:
            norm_method = st.radio(
                "Method",
                ["Min-Max (0-1)", "Z-Score", "% of Max"],
                key=f"explore_norm_method_{sheet_dataset_key}_{active_sheet}",
                help="Min-Max: 0-1 scale\nZ-Score: standardized\n% of Max: percentage"
            )
    
    with viz_col:
        if not chart_metrics:
            st.info("👈 Select metrics from the configuration panel")
        else:
            df_agg = aggregate_data(filtered_df, agg_level, agg_method.lower())
            
            if enable_normalization:
                norm_method_map = {
                    "Min-Max (0-1)": "minmax",
                    "Z-Score": "zscore",
                    "% of Max": "pct_max"
                }
                df_agg = DataProcessor.normalize_data(
                    df_agg,
                    chart_metrics,
                    method=norm_method_map[norm_method]
                )
                plot_metrics = [f"{m}_norm" for m in chart_metrics]
                chart_title_suffix = f" ({norm_method})"
            else:
                plot_metrics = chart_metrics
                chart_title_suffix = ""
            
            fig = go.Figure()
            
            for idx, metric in enumerate(plot_metrics):
                original_metric = metric.replace('_norm', '')
                
                if chart_type == "Line":
                    fig.add_trace(go.Scatter(
                        x=df_agg['Period'],
                        y=df_agg[metric],
                        name=original_metric,
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
                elif chart_type == "Bar":
                    fig.add_trace(go.Bar(
                        x=df_agg['Period'],
                        y=df_agg[metric],
                        name=original_metric
                    ))
                elif chart_type == "Area":
                    fig.add_trace(go.Scatter(
                        x=df_agg['Period'],
                        y=df_agg[metric],
                        name=original_metric,
                        fill='tonexty' if idx > 0 else 'tozeroy',
                        mode='lines'
                    ))
                elif chart_type == "Scatter":
                    fig.add_trace(go.Scatter(
                        x=df_agg['Period'],
                        y=df_agg[metric],
                        name=original_metric,
                        mode='markers',
                        marker=dict(size=10, opacity=0.6)
                    ))
                
                if show_trend and chart_type == "Line":
                    trend = DataProcessor.calculate_trend(df_agg, metric)
                    if trend:
                        fig.add_trace(go.Scatter(
                            x=df_agg['Period'],
                            y=trend['trend_line'],
                            name=f"{original_metric} (Trend, R²={trend['r_squared']:.3f})",
                            mode='lines',
                            line=dict(dash='dot', width=2),
                            opacity=0.6
                        ))
            
            title = f"{agg_method} by {agg_level}{chart_title_suffix}"
            fig.update_layout(
                title=title,
                height=600,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📊 View Period Statistics"):
                stats_df = df_agg[['Period'] + chart_metrics].set_index('Period')
                st.dataframe(stats_df.describe().T, use_container_width=True)


# TAB 3: PIVOT ANALYSIS
with tabs[2]:
    st.markdown("""
    <div class="section-header">
        <h2>🔄 Pivot Analysis</h2>
        <p>Multi-dimensional data aggregation with totals and percentages</p>
    </div>
    """, unsafe_allow_html=True)
    
    result = render_worksheet_selector(selected_key, "pivot", show_dataset_selector=True)
    if isinstance(result, tuple):
        active_sheet, sheet_dataset_key = result
    else:
        active_sheet = result
        sheet_dataset_key = selected_key
    
    sheet_ds = DatasetManager.get_dataset(sheet_dataset_key)
    filtered_df, filters, start, end = render_filter_controls(sheet_dataset_key, "pivot", active_sheet)
    
    if filtered_df.empty:
        st.warning("📊 No data matches the current filters")
        st.stop()
    
    # Get stored results for this sheet
    sheet_results = DatasetManager.get_sheet_results(sheet_dataset_key, "pivot", active_sheet)
    
    # SIDE-BY-SIDE LAYOUT: Configuration (Left) | Results (Right)
    config_col, result_col = st.columns([1, 2.5])
    
    with config_col:
        st.markdown("### ⚙️ Pivot Configuration")
        
        # Load saved configuration or use defaults
        saved_config = sheet_results.get('pivot_config', {})
        
        # Pivot Structure
        st.markdown("**📊 Structure**")
        available_rows = sheet_ds['selected_identifiers'] + (['Day'] if sheet_ds['has_dates'] else [])
        pivot_rows = st.multiselect(
            "Rows",
            available_rows,
            default=saved_config.get('rows', []),
            key=f"pivot_rows_{sheet_dataset_key}_{active_sheet}",
            help="Dimensions shown vertically"
        )
        
        pivot_cols = st.multiselect(
            "Columns",
            sheet_ds['selected_identifiers'],
            default=saved_config.get('cols', []),
            key=f"pivot_cols_{sheet_dataset_key}_{active_sheet}",
            help="Dimensions shown horizontally (optional)"
        )
        
        pivot_vals = st.multiselect(
            "Values",
            sheet_ds['selected_metrics'],
            default=saved_config.get('vals', []),
            key=f"pivot_vals_{sheet_dataset_key}_{active_sheet}",
            help="Metrics to aggregate"
        )
        
        st.divider()
        
        # Aggregation Settings
        st.markdown("**⚙️ Settings**")
        pivot_agg = st.selectbox(
            "Aggregation Method",
            ["Sum", "Mean", "Count", "Min", "Max"],
            index=["Sum", "Mean", "Count", "Min", "Max"].index(saved_config.get('agg', 'Sum')),
            key=f"pivot_agg_{sheet_dataset_key}_{active_sheet}"
        )
        
        show_grand_totals = st.checkbox(
            "Show Grand Totals",
            value=saved_config.get('totals', True),
            key=f"pivot_totals_{sheet_dataset_key}_{active_sheet}",
            help="Add row and column totals"
        )
        
        show_percentages = st.checkbox(
            "Show Percentages",
            value=saved_config.get('show_pct', False),
            key=f"pivot_pct_{sheet_dataset_key}_{active_sheet}",
            help="Display values as percentages"
        )
        
        if show_percentages:
            pct_type = st.radio(
                "Percentage Of",
                ["Column Total", "Row Total", "Grand Total"],
                index=["Column Total", "Row Total", "Grand Total"].index(saved_config.get('pct_type', 'Column Total')),
                key=f"pivot_pct_type_{sheet_dataset_key}_{active_sheet}",
                help="Calculate % relative to selected total"
            )
        else:
            pct_type = "Column Total"
        
        st.divider()
        
        # Generate Button
        can_generate = len(pivot_rows) > 0 and len(pivot_vals) > 0
        
        if st.button(
            "🔄 Generate Pivot Table", 
            type="primary", 
            disabled=not can_generate, 
            use_container_width=True,
            key=f"pivot_gen_{sheet_dataset_key}_{active_sheet}"
        ):
            # Save configuration
            current_config = {
                'rows': pivot_rows,
                'cols': pivot_cols,
                'vals': pivot_vals,
                'agg': pivot_agg,
                'totals': show_grand_totals,
                'show_pct': show_percentages,
                'pct_type': pct_type
            }
            
            # Generate pivot table
            agg_map = {'Sum': 'sum', 'Mean': 'mean', 'Count': 'count', 'Min': 'min', 'Max': 'max'}
            
            if show_grand_totals:
                pct_type_map = {
                    "Column Total": "column",
                    "Row Total": "row",
                    "Grand Total": "grand"
                }
                
                pivot_result, pct_pivot = DataProcessor.create_pivot_with_totals(
                    filtered_df,
                    index=pivot_rows,
                    columns=pivot_cols if pivot_cols else None,
                    values=pivot_vals,
                    aggfunc=agg_map[pivot_agg],
                    show_percentages=show_percentages,
                    pct_type=pct_type_map.get(pct_type, 'column') if show_percentages else None
                )
            else:
                if not pivot_cols:
                    pivot_result = filtered_df.groupby(pivot_rows).agg({v: agg_map[pivot_agg] for v in pivot_vals}).reset_index()
                else:
                    pivot_result = pd.pivot_table(
                        filtered_df,
                        values=pivot_vals,
                        index=pivot_rows,
                        columns=pivot_cols,
                        aggfunc=agg_map[pivot_agg],
                        fill_value=0
                    ).reset_index()
                pct_pivot = None
            
            # Store results and configuration
            DatasetManager.update_sheet_results(
                sheet_dataset_key,
                "pivot",
                active_sheet,
                pivot_generated=True,
                pivot_config=current_config,
                pivot_result=pivot_result,
                pct_pivot=pct_pivot
            )
            
            st.rerun()
        
        if not can_generate:
            st.caption("⚠️ Select at least 1 row and 1 value")
        
        # Show current configuration status and clear button
        if sheet_results.get('pivot_generated', False):
            st.success("✅ Results saved for this sheet")
            if st.button("🗑️ Clear Results", use_container_width=True, key=f"clear_pivot_{sheet_dataset_key}_{active_sheet}"):
                DatasetManager.update_sheet_results(
                    sheet_dataset_key,
                    "pivot",
                    active_sheet,
                    pivot_generated=False,
                    pivot_result=None,
                    pct_pivot=None
                )
                st.rerun()
    
    with result_col:
        # RESULTS DISPLAY
        if sheet_results.get('pivot_generated', False):
            pivot_result = sheet_results.get('pivot_result')
            pct_pivot = sheet_results.get('pct_pivot')
            saved_config = sheet_results.get('pivot_config', {})
            
            if pivot_result is not None:
                # Display results
                result_tabs = st.tabs(["📊 Values"] + (["📈 Percentages"] if pct_pivot is not None else []))
                
                with result_tabs[0]:
                    st.markdown("**Pivot Table Results**")
                    styled_pivot = pivot_result.style.format({
                        col: lambda x: format_value(x, col) if isinstance(x, (int, float)) else x
                        for col in pivot_result.columns if pivot_result[col].dtype in [np.int64, np.float64]
                    })
                    
                    if saved_config.get('totals', True) and 'Grand Total' in pivot_result.index:
                        styled_pivot = styled_pivot.apply(
                            lambda x: ['background-color: #FFE7C2; font-weight: bold' 
                                      if x.name == 'Grand Total' else '' for i in x],
                            axis=1
                        )
                    
                    st.dataframe(styled_pivot, use_container_width=True, height=500)
                    
                    csv_data = pivot_result.to_csv(index=True)
                    st.download_button(
                        "📥 Download Values",
                        csv_data,
                        f"pivot_values_{datetime.now():%Y%m%d_%H%M%S}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                if pct_pivot is not None:
                    with result_tabs[1]:
                        st.markdown("**Percentage View**")
                        styled_pct = pct_pivot.style.format({
                            col: lambda x: format_value(x, '', as_percentage=True) if isinstance(x, (int, float)) else x
                            for col in pct_pivot.columns if pct_pivot[col].dtype in [np.int64, np.float64]
                        })
                        
                        if 'Grand Total' in pct_pivot.index:
                            styled_pct = styled_pct.apply(
                                lambda x: ['background-color: #FFE7C2; font-weight: bold' 
                                          if x.name == 'Grand Total' else '' for i in x],
                                axis=1
                            )
                        
                        st.dataframe(styled_pct, use_container_width=True, height=500)
                        
                        csv_pct = pct_pivot.to_csv(index=True)
                        st.download_button(
                            "📥 Download Percentages",
                            csv_pct,
                            f"pivot_percentages_{datetime.now():%Y%m%d_%H%M%S}.csv",
                            "text/csv",
                            use_container_width=True
                        )
            else:
                st.warning("⚠️ Results were cleared. Please regenerate the pivot table.")
        else:
            st.info("👈 Configure pivot settings on the left and click 'Generate Pivot Table'")
            
            with st.expander("💡 Quick Guide"):
                st.markdown("""
                **How to create a pivot table:**
                
                1. **Rows**: Select dimensions to group vertically (e.g., Region, Product)
                2. **Columns**: Optional horizontal dimension (e.g., Month)
                3. **Values**: Select metrics to aggregate (e.g., Sales, Revenue)
                4. **Aggregation**: Choose how to combine values (Sum, Mean, etc.)
                5. **Grand Totals**: Add row and column totals
                6. **Percentages**: View data as % of total
                
                **Example**: Rows=[Region], Columns=[Product Category], Values=[Sales]
                
                Creates a matrix showing sales by region and category.
                
                **💡 Results are saved per sheet** - switch between sheets without losing work!
                """)



# TAB 4: CORRELATION
with tabs[3]:
    st.markdown("""
    <div class="section-header">
        <h2>🔗 Correlation Analysis</h2>
        <p>Identify relationships between metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    result = render_worksheet_selector(selected_key, "correlation", show_dataset_selector=True)
    if isinstance(result, tuple):
        active_sheet, sheet_dataset_key = result
    else:
        active_sheet = result
        sheet_dataset_key = selected_key
    
    sheet_ds = DatasetManager.get_dataset(sheet_dataset_key)
    
    if len(sheet_ds['selected_metrics']) < 2:
        st.warning("⚠️ Need at least 2 metrics for correlation analysis")
        st.stop()
    
    filtered_df, filters, start, end = render_filter_controls(sheet_dataset_key, "correlation", active_sheet)
    
    if filtered_df.empty:
        st.warning("📊 No data matches the current filters")
        st.stop()
    
    config_col, viz_col = st.columns([1, 2.5])
    
    with config_col:
        st.markdown("### ⚙️ Settings")
        corr_metrics = st.multiselect(
            "Metrics",
            sheet_ds['selected_metrics'],
            default=sheet_ds['selected_metrics'][:min(5, len(sheet_ds['selected_metrics']))],
            key=f"corr_metrics_{sheet_dataset_key}_{active_sheet}"
        )
        corr_method = st.selectbox(
            "Method",
            ["Pearson", "Spearman", "Kendall"],
            key=f"corr_method_{sheet_dataset_key}_{active_sheet}",
            help="Pearson: linear\nSpearman: monotonic\nKendall: rank"
        )
        
        if len(corr_metrics) >= 2:
            min_corr = st.slider(
                "Highlight ≥",
                0.0, 1.0, 0.5, 0.05,
                key=f"min_corr_{sheet_dataset_key}_{active_sheet}"
            )
    
    with viz_col:
        if len(corr_metrics) < 2:
            st.info("👈 Select at least 2 metrics")
        else:
            # Tabs for regular and lag correlation
            corr_tab1, corr_tab2 = st.tabs(["📊 Standard Correlation", "⏱️ Lag Correlation"])
            
            with corr_tab1:
                corr_matrix = filtered_df[corr_metrics].corr(method=corr_method.lower())
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_metrics,
                    y=corr_metrics,
                    colorscale='RdBu_r',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                
                fig.update_layout(
                    title=f"{corr_method} Correlation Matrix",
                    height=600,
                    xaxis={'side': 'bottom'},
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("🔍 Strong Correlations"):
                    strong_corrs = []
                    for i in range(len(corr_matrix)):
                        for j in range(i+1, len(corr_matrix)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) >= min_corr:
                                strong_corrs.append({
                                    'Metric 1': corr_metrics[i],
                                    'Metric 2': corr_metrics[j],
                                    'Correlation': corr_val,
                                    'Strength': 'Strong' if abs(corr_val) >= 0.7 else 'Moderate'
                                })
                    
                    if strong_corrs:
                        strong_df = pd.DataFrame(strong_corrs).sort_values('Correlation', key=abs, ascending=False)
                        st.dataframe(strong_df, use_container_width=True, hide_index=True)
                    else:
                        st.info(f"No correlations ≥ {min_corr:.2f}")
            
            with corr_tab2:
                st.markdown("### Lag Correlation Analysis")
                st.info("Analyze how one metric (Leading/X) predicts another metric (Lagging/Y) in future time periods. Useful for identifying delayed effects - e.g., how impressions this week affect sessions next week.")
                
                lag_col1, lag_col2 = st.columns(2)
                with lag_col1:
                    lag_metric_x = st.selectbox(
                        "Leading Metric (X)",
                        corr_metrics,
                        key=f"lag_x_{sheet_dataset_key}_{active_sheet}"
                    )
                with lag_col2:
                    lag_metric_y = st.selectbox(
                        "Lagging Metric (Y)",
                        [m for m in corr_metrics if m != lag_metric_x],
                        key=f"lag_y_{sheet_dataset_key}_{active_sheet}"
                    )
                
                max_lag = st.slider(
                    "Maximum Lag Periods",
                    1, min(20, len(filtered_df) // 2), 7,
                    key=f"max_lag_{sheet_dataset_key}_{active_sheet}",
                    help="Number of time periods to shift for correlation analysis"
                )
                
                if lag_metric_x and lag_metric_y:
                    # Calculate lag correlations
                    lag_correlations = []
                    for lag in range(0, max_lag + 1):
                        if lag == 0:
                            corr_val = filtered_df[lag_metric_x].corr(filtered_df[lag_metric_y], method=corr_method.lower())
                        else:
                            # Shift Y metric BACKWARD (negative shift) to check if X predicts future Y
                            # This aligns current X values with future Y values
                            y_shifted = filtered_df[lag_metric_y].shift(-lag)
                            corr_val = filtered_df[lag_metric_x].corr(y_shifted, method=corr_method.lower())
                        
                        lag_correlations.append({
                            'Lag': lag,
                            'Correlation': corr_val
                        })
                    
                    lag_df = pd.DataFrame(lag_correlations)
                    
                    # Plot lag correlation
                    fig_lag = go.Figure()
                    fig_lag.add_trace(go.Scatter(
                        x=lag_df['Lag'],
                        y=lag_df['Correlation'],
                        mode='lines+markers',
                        name='Correlation',
                        line=dict(color='#FFBD59', width=3),
                        marker=dict(size=10, color='#41C185')
                    ))
                    
                    # Add zero line
                    fig_lag.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    # Highlight max correlation
                    max_corr_idx = lag_df['Correlation'].abs().idxmax()
                    max_corr_lag = lag_df.loc[max_corr_idx, 'Lag']
                    max_corr_val = lag_df.loc[max_corr_idx, 'Correlation']
                    
                    fig_lag.add_trace(go.Scatter(
                        x=[max_corr_lag],
                        y=[max_corr_val],
                        mode='markers',
                        name='Max Correlation',
                        marker=dict(size=15, color='#E74C3C', symbol='star')
                    ))
                    
                    fig_lag.update_layout(
                        title=f"Lag Correlation: {lag_metric_x} → {lag_metric_y}",
                        xaxis_title="Lag (periods)",
                        yaxis_title=f"{corr_method} Correlation",
                        height=500,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_lag, use_container_width=True)
                    
                    # Display insights
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Correlation", f"{max_corr_val:.3f}", f"at lag {int(max_corr_lag)}")
                    with col2:
                        st.metric("Zero Lag Correlation", f"{lag_df.loc[0, 'Correlation']:.3f}")
                    with col3:
                        improvement = abs(max_corr_val) - abs(lag_df.loc[0, 'Correlation'])
                        st.metric("Improvement", f"{improvement:.3f}", 
                                 f"{'↑' if improvement > 0 else '↓'} vs lag 0")
                    
                    with st.expander("📊 Lag Correlation Table"):
                        st.dataframe(lag_df.style.background_gradient(subset=['Correlation'], cmap='RdBu_r', vmin=-1, vmax=1),
                                   use_container_width=True, hide_index=True)
                    
                    # Interpretation
                    st.markdown("### 💡 Interpretation")
                    if max_corr_lag == 0:
                        st.success(f"**Strongest correlation at lag 0**: {lag_metric_x} and {lag_metric_y} are contemporaneously correlated (no time delay).")
                    else:
                        st.success(f"**Strongest correlation at lag {int(max_corr_lag)}**: Changes in {lag_metric_x} are associated with changes in {lag_metric_y} after {int(max_corr_lag)} period(s).")
                        if max_corr_val > 0:
                            st.info(f"**Positive correlation**: When {lag_metric_x} increases, {lag_metric_y} tends to increase {int(max_corr_lag)} period(s) later.")
                        else:
                            st.info(f"**Negative correlation**: When {lag_metric_x} increases, {lag_metric_y} tends to decrease {int(max_corr_lag)} period(s) later.")

# TAB 5: CLUSTERING
with tabs[4]:
    st.markdown("""
    <div class="section-header">
        <h2>🎯 Clustering Analysis</h2>
        <p>Discover patterns and segments in your data</p>
    </div>
    """, unsafe_allow_html=True)
    
    result = render_worksheet_selector(selected_key, "clustering", show_dataset_selector=True)
    if isinstance(result, tuple):
        active_sheet, sheet_dataset_key = result
    else:
        active_sheet = result
        sheet_dataset_key = selected_key
    
    sheet_ds = DatasetManager.get_dataset(sheet_dataset_key)
    
    if len(sheet_ds['selected_metrics']) < 2:
        st.warning("⚠️ Need at least 2 metrics for clustering")
        st.stop()
    
    filtered_df, filters, start, end = render_filter_controls(sheet_dataset_key, "clustering", active_sheet)
    
    if filtered_df.empty:
        st.warning("📊 No data matches the current filters")
        st.stop()
    
    config_col, viz_col = st.columns([1, 2.5])
    
    with config_col:
        st.markdown("### ⚙️ Settings")
        cluster_features = st.multiselect(
            "Features",
            sheet_ds['selected_metrics'],
            default=sheet_ds['selected_metrics'][:min(3, len(sheet_ds['selected_metrics']))],
            key=f"cluster_features_{sheet_dataset_key}_{active_sheet}"
        )
        
        if len(cluster_features) >= 2:
            algo = st.selectbox(
                "Algorithm",
                ["K-Means", "DBSCAN"],
                key=f"cluster_algo_{sheet_dataset_key}_{active_sheet}"
            )
            
            if algo == "K-Means":
                n_clusters = st.slider(
                    "Clusters",
                    2, 10, 3,
                    key=f"n_clusters_{sheet_dataset_key}_{active_sheet}"
                )
            else:
                eps = st.slider(
                    "Epsilon",
                    0.1, 5.0, 0.5, 0.1,
                    key=f"eps_{sheet_dataset_key}_{active_sheet}"
                )
                min_samples = st.slider(
                    "Min Samples",
                    2, 10, 5,
                    key=f"min_samp_{sheet_dataset_key}_{active_sheet}"
                )
            
            viz_x = st.selectbox(
                "X-axis",
                cluster_features,
                key=f"viz_x_{sheet_dataset_key}_{active_sheet}"
            )
            viz_y = st.selectbox(
                "Y-axis",
                cluster_features,
                index=min(1, len(cluster_features)-1),
                key=f"viz_y_{sheet_dataset_key}_{active_sheet}"
            )
            
            if st.button("▶️ Run Analysis", type="primary", use_container_width=True, key=f"cluster_run_{sheet_dataset_key}_{active_sheet}"):
                st.session_state[f'clustering_run_{sheet_dataset_key}_{active_sheet}'] = True
                st.rerun()
    
    with viz_col:
        if len(cluster_features) < 2:
            st.info("👈 Select at least 2 features")
        elif st.session_state.get(f'clustering_run_{sheet_dataset_key}_{active_sheet}', False):
            try:
                from sklearn.cluster import KMeans, DBSCAN
                from sklearn.preprocessing import StandardScaler
                
                cluster_data = filtered_df[cluster_features].dropna()
                scaler = StandardScaler()
                scaled = scaler.fit_transform(cluster_data)
                
                if algo == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                else:
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                
                labels = model.fit_predict(scaled)
                cluster_data['Cluster'] = labels
                
                fig = px.scatter(
                    cluster_data,
                    x=viz_x,
                    y=viz_y,
                    color='Cluster',
                    title=f"{algo} Clustering Results",
                    height=500,
                    color_continuous_scale='viridis' if algo == "K-Means" else None
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📊 Cluster Statistics"):
                    cluster_stats = cluster_data.groupby('Cluster')[cluster_features].agg(['mean', 'count'])
                    st.dataframe(cluster_stats, use_container_width=True)
                
                with st.expander("📊 Cluster Distribution"):
                    cluster_counts = cluster_data['Cluster'].value_counts().sort_index()
                    fig_dist = go.Figure(data=[go.Bar(
                        x=cluster_counts.index.astype(str),
                        y=cluster_counts.values,
                        marker_color='#FFBD59'
                    )])
                    fig_dist.update_layout(
                        title="Data Points per Cluster",
                        xaxis_title="Cluster",
                        yaxis_title="Count",
                        height=300
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
            except ImportError:
                st.error("❌ Install scikit-learn: `pip install scikit-learn`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👈 Configure and click 'Run Analysis'")

# TAB 6: INSIGHTS
with tabs[5]:
    st.markdown("""
    <div class="section-header">
        <h2>📋 Insights & Comments</h2>
        <p>Manage your analysis notes and observations</p>
    </div>
    """, unsafe_allow_html=True)
    
    comments_df, error = CommentManager.load_comments()
    
    if error:
        st.error(f"⚠️ {error}")
    
    if not comments_df.empty:
        st.markdown(f"**{len(comments_df)} saved insights**")
        
        csv_export, _ = CommentManager.export_comments()
        if csv_export:
            st.download_button(
                "📥 Export All",
                csv_export,
                f"insights_{datetime.now():%Y%m%d}.csv",
                "text/csv"
            )
        
        st.divider()
        
        comments_df = comments_df.sort_values('timestamp', ascending=False)
        
        for _, row in comments_df.iterrows():
            with st.expander(f"💬 {row['tab_name']} • {row['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.markdown(f"_{row['comment_text']}_")
                
                if row['context_data'] and row['context_data'] != 'null':
                    try:
                        context = json.loads(row['context_data'])
                        st.caption("**Context:** " + str(context))
                    except:
                        pass
                
                if st.button("🗑️ Delete", key=f"del_{row['id']}"):
                    CommentManager.delete_comment(row['id'])
                    st.rerun()
    else:
        st.info("📝 No insights yet. Add notes from any analysis tab!")

# TAB 6: FEATURE IMPORTANCE
with tabs[5]:
    st.markdown("""
    <div class="section-header">
        <h2>🎯 Feature Importance Analysis</h2>
        <p>Identify which features best explain your target variable using XGBoost</p>
    </div>
    """, unsafe_allow_html=True)
    
    result = render_worksheet_selector(selected_key, "feature_importance", show_dataset_selector=True)
    if isinstance(result, tuple):
        active_sheet, sheet_dataset_key = result
    else:
        active_sheet = result
        sheet_dataset_key = selected_key
    
    sheet_ds = DatasetManager.get_dataset(sheet_dataset_key)
    
    if len(sheet_ds['selected_metrics']) < 2:
        st.warning("⚠️ Need at least 2 metrics for feature importance analysis")
        st.stop()
    
    filtered_df, filters, start, end = render_filter_controls(sheet_dataset_key, "feature_importance", active_sheet)
    
    if filtered_df.empty:
        st.warning("📊 No data matches the current filters")
        st.stop()
    
    config_col, viz_col = st.columns([1, 2.5])
    
    with config_col:
        st.markdown("### ⚙️ Settings")
        
        # Select target variable (Y)
        target_var = st.selectbox(
            "Target Variable (Y)",
            sheet_ds['selected_metrics'],
            key=f"target_var_{sheet_dataset_key}_{active_sheet}",
            help="The variable you want to predict/explain"
        )
        
        # Select feature variables (X)
        available_features = [m for m in sheet_ds['selected_metrics'] if m != target_var]
        feature_vars = st.multiselect(
            "Feature Variables (X)",
            available_features,
            default=available_features[:min(10, len(available_features))],
            key=f"feature_vars_{sheet_dataset_key}_{active_sheet}",
            help="Variables that might explain the target"
        )
        
        if len(feature_vars) >= 1:
            st.markdown("#### Model Settings")
            
            auto_tune = st.checkbox(
                "🔧 Auto-tune Hyperparameters",
                value=False,
                key=f"auto_tune_{sheet_dataset_key}_{active_sheet}",
                help="Automatically find best hyperparameters using GridSearchCV (takes longer)"
            )
            
            if not auto_tune:
                max_depth = st.slider(
                    "Max Depth",
                    2, 10, 5,
                    key=f"max_depth_{sheet_dataset_key}_{active_sheet}",
                    help="Maximum tree depth"
                )
                
                n_estimators = st.slider(
                    "Number of Trees",
                    10, 200, 100, 10,
                    key=f"n_estimators_{sheet_dataset_key}_{active_sheet}",
                    help="Number of boosting rounds"
                )
                
                learning_rate = st.slider(
                    "Learning Rate",
                    0.01, 0.3, 0.1, 0.01,
                    key=f"learning_rate_{sheet_dataset_key}_{active_sheet}",
                    help="Step size shrinkage to prevent overfitting"
                )
            else:
                st.info("⚙️ Will search for optimal: max_depth, n_estimators, learning_rate")
                max_depth = None
                n_estimators = None
                learning_rate = None
            
            importance_type = st.selectbox(
                "Importance Type",
                ["gain", "weight", "cover"],
                key=f"importance_type_{sheet_dataset_key}_{active_sheet}",
                help="gain: average gain\nweight: number of times used\ncover: average coverage"
            )
            
            run_analysis = st.button("▶️ Run Analysis", type="primary", use_container_width=True, 
                        key=f"feat_imp_run_{sheet_dataset_key}_{active_sheet}")
            
            # Explanation of importance types
            with st.expander("ℹ️ What do importance types mean?"):
                st.markdown("""
                **Importance Type Explanations:**
                
                - **Gain** (Recommended)
                  - Measures the average improvement in accuracy brought by a feature
                  - Higher gain = feature provides more valuable splits
                  - Best for understanding which features improve predictions most
                
                - **Weight**
                  - Counts how many times a feature is used in the model
                  - Higher weight = feature is used more frequently
                  - Useful for understanding feature usage frequency
                
                - **Cover**
                  - Measures the average coverage (number of samples affected) by splits using the feature
                  - Higher cover = feature affects more data points
                  - Useful for understanding feature's breadth of impact
                
                **Which to choose?**
                - Use **Gain** for most cases - it shows which features truly improve predictions
                - Use **Weight** if you want to see which features the model relies on most
                - Use **Cover** if you want to understand which features affect the most samples
                """)
    
    with viz_col:
        if len(feature_vars) < 1:
            st.info("👈 Select at least 1 feature variable")
        elif run_analysis:
            try:
                import xgboost as xgb
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                
                with st.spinner("Training XGBoost model..."):
                    # Prepare data
                    X = filtered_df[feature_vars].fillna(0)
                    y = filtered_df[target_var].fillna(0)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Train XGBoost model
                    tuning_possible = auto_tune
                    cv_folds = None
                    if auto_tune:
                        cv_folds = min(3, len(X_train))
                        if cv_folds < 2:
                            st.warning("⚠️ Dataset too small for cross-validated hyperparameter search. Running default model instead.")
                            tuning_possible = False
                    
                    if tuning_possible:
                        from sklearn.model_selection import GridSearchCV
                        
                        st.info("🔍 Searching for optimal hyperparameters... This may take a moment.")
                        
                        # Define parameter grid
                        param_grid = {
                            'max_depth': [3, 5, 7],
                            'n_estimators': [50, 100, 150],
                            'learning_rate': [0.01, 0.1, 0.2]
                        }
                        
                        # Create base model
                        base_model = xgb.XGBRegressor(
                            random_state=42,
                            importance_type=importance_type
                        )
                        
                        # Grid search with cross-validation
                        grid_search = GridSearchCV(
                            base_model,
                            param_grid,
                            cv=cv_folds,
                            scoring='r2',
                            n_jobs=-1,
                            verbose=0
                        )
                        grid_search.fit(X_train, y_train)
                        
                        # Use best model
                        model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        
                        # Display best parameters
                        st.success(f"✅ Best parameters found: max_depth={best_params['max_depth']}, "
                                  f"n_estimators={best_params['n_estimators']}, "
                                  f"learning_rate={best_params['learning_rate']:.3f}")
                    else:
                        model = xgb.XGBRegressor(
                            max_depth=max_depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            random_state=42,
                            importance_type=importance_type
                        )
                        model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Get feature importance
                    importance_dict = model.get_booster().get_score(importance_type=importance_type)
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame([
                        {'Feature': k.replace('f', feature_vars[int(k[1:])]) if k.startswith('f') else k, 
                         'Importance': v}
                        for k, v in importance_dict.items()
                    ]).sort_values('Importance', ascending=False)
                    
                    # Map feature names correctly
                    feature_map = {f'f{i}': feature_vars[i] for i in range(len(feature_vars))}
                    importance_df['Feature'] = importance_df['Feature'].map(
                        lambda x: feature_map.get(x, x)
                    )
                    
                    # Build feature-importance plot object
                    fig_imp = go.Figure()
                    fig_imp.add_trace(go.Bar(
                        y=importance_df['Feature'][::-1],
                        x=importance_df['Importance'][::-1],
                        orientation='h',
                        marker=dict(
                            color=importance_df['Importance'][::-1],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Importance")
                        ),
                        text=importance_df['Importance'][::-1].round(2),
                        textposition='auto'
                    ))
                    
                    fig_imp.update_layout(
                        title=f"Feature Importance for {target_var}",
                        xaxis_title=f"Importance ({importance_type})",
                        yaxis_title="Features",
                        height=max(400, len(importance_df) * 30),
                        template='plotly_white',
                        showlegend=False
                    )
                    
                    # Actual vs Predicted plot object
                    fig_pred = go.Figure()
                    
                    fig_pred.add_trace(go.Scatter(
                        x=y_test,
                        y=y_pred,
                        mode='markers',
                        name='Predictions',
                        marker=dict(
                            size=8,
                            color='#41C185',
                            opacity=0.6
                        )
                    ))
                    
                    # Add perfect prediction line
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig_pred.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='#E74C3C', dash='dash')
                    ))
                    
                    fig_pred.update_layout(
                        title=f"Actual vs Predicted {target_var}",
                        xaxis_title=f"Actual {target_var}",
                        yaxis_title=f"Predicted {target_var}",
                        height=500,
                        template='plotly_white'
                    )
                    
                    # Contribution-driven interpretability
                    contribution_summary = None
                    contrib_df = None
                    try:
                        dtest = xgb.DMatrix(X_test, feature_names=feature_vars)
                        contrib_array = model.get_booster().predict(dtest, pred_contribs=True)
                        contrib_df = pd.DataFrame(contrib_array, columns=feature_vars + ['bias'])
                        
                        mean_signed = contrib_df[feature_vars].mean()
                        mean_abs = contrib_df[feature_vars].abs().mean()
                        
                        corr_map = {}
                        for feat in feature_vars:
                            feat_vals = X_test[feat]
                            if np.std(feat_vals) == 0 or np.std(y_test) == 0:
                                corr_map[feat] = np.nan
                            else:
                                corr_map[feat] = float(np.corrcoef(feat_vals, y_test)[0, 1])
                        
                        contribution_summary = pd.DataFrame({
                            'Feature': feature_vars,
                            'Avg Contribution': mean_signed.values,
                            'Avg |Contribution|': mean_abs.values,
                            'Correlation': [corr_map[feat] for feat in feature_vars]
                        })
                        
                        def describe_direction(value):
                            if value > 0:
                                return "⬆ Higher values push predictions up"
                            if value < 0:
                                return "⬇ Higher values pull predictions down"
                            return "Neutral impact"
                        
                        contribution_summary['Impact'] = contribution_summary['Avg Contribution'].apply(describe_direction)
                        contribution_summary = contribution_summary.sort_values('Avg |Contribution|', ascending=False).reset_index(drop=True)
                    except Exception:
                        contribution_summary = None
                        contrib_df = None
                    
                    # Pre-compute percentages for downstream use
                    if 'Percentage' not in importance_df.columns and not importance_df.empty:
                        importance_df['Percentage'] = (
                            importance_df['Importance'] / importance_df['Importance'].sum() * 100
                        ).round(2)
                        importance_df['Cumulative %'] = importance_df['Percentage'].cumsum().round(2)
                    
                    # Organize visual outputs into tabs
                    result_tabs = st.tabs([
                        "📊 Model Performance",
                        "🎯 Feature Importance",
                        "📈 Actual vs Predicted",
                        "🧠 Interpretability"
                    ])
                    
                    with result_tabs[0]:
                        st.markdown("### 📊 Model Performance")
                        perf_col1, perf_col2, perf_col3 = st.columns(3)
                        with perf_col1:
                            st.metric("R² Score", f"{r2:.3f}", 
                                     help="Proportion of variance explained (higher is better)")
                        with perf_col2:
                            st.metric("RMSE", f"{rmse:.2f}",
                                     help="Root Mean Squared Error (lower is better)")
                        with perf_col3:
                            st.metric("MAE", f"{mae:.2f}",
                                     help="Mean Absolute Error (lower is better)")
                    
                    with result_tabs[1]:
                        st.markdown("### 🎯 Feature Importance")
                        st.plotly_chart(fig_imp, use_container_width=True)
                        
                        with st.expander("📋 Detailed Table"):
                            st.dataframe(
                                importance_df.style.background_gradient(
                                    subset=['Importance'], cmap='YlOrRd'
                                ),
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    with result_tabs[2]:
                        st.markdown("### 📈 Actual vs Predicted")
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    with result_tabs[3]:
                        st.markdown("### 🧭 Contribution Breakdown")
                        if contribution_summary is not None and not contribution_summary.empty:
                            st.caption("Average SHAP-like contributions (via `pred_contribs`) on the validation split explain how each feature shifts predictions.")
                            display_contrib = contribution_summary.copy()
                            display_contrib['Avg Contribution'] = display_contrib['Avg Contribution'].map(lambda v: f"{v:.4f}")
                            display_contrib['Avg |Contribution|'] = display_contrib['Avg |Contribution|'].map(lambda v: f"{v:.4f}")
                            display_contrib['Correlation'] = display_contrib['Correlation'].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
                            st.dataframe(
                                display_contrib[['Feature', 'Avg Contribution', 'Avg |Contribution|', 'Correlation', 'Impact']],
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            st.markdown("### 🔍 Feature Effect Explorer")
                            explorer_feature = st.selectbox(
                                "Inspect contribution pattern",
                                contribution_summary['Feature'],
                                key=f"feature_effect_{sheet_dataset_key}_{active_sheet}"
                            )
                            
                            if contrib_df is not None and explorer_feature in contrib_df.columns:
                                explorer_df = pd.DataFrame({
                                    explorer_feature: X_test[explorer_feature],
                                    'Contribution': contrib_df[explorer_feature],
                                    target_var: y_test
                                }).dropna()
                                
                                if not explorer_df.empty:
                                    fig_effect = go.Figure()
                                    fig_effect.add_trace(go.Scatter(
                                        x=explorer_df[explorer_feature],
                                        y=explorer_df['Contribution'],
                                        mode='markers',
                                        marker=dict(
                                            size=8,
                                            color=explorer_df[target_var],
                                            colorscale='RdBu',
                                            showscale=True,
                                            colorbar=dict(title=target_var)
                                        ),
                                        name='Feature Contribution'
                                    ))
                                    
                                    fig_effect.update_layout(
                                        title=f"{explorer_feature} vs Contribution",
                                        xaxis_title=explorer_feature,
                                        yaxis_title="Contribution to prediction",
                                        template='plotly_white'
                                    )
                                    
                                    st.plotly_chart(fig_effect, use_container_width=True)
                                
                                corr_val = contribution_summary.loc[
                                    contribution_summary['Feature'] == explorer_feature, 'Correlation'
                                ].iloc[0]
                                avg_contrib = contribution_summary.loc[
                                    contribution_summary['Feature'] == explorer_feature, 'Avg Contribution'
                                ].iloc[0]
                                
                                direction_text = "increase" if avg_contrib > 0 else "decrease" if avg_contrib < 0 else "not consistently move"
                                corr_text = f"{corr_val:.2f}" if pd.notna(corr_val) else "N/A"
                                st.info(
                                    f"On average, higher **{explorer_feature}** values {direction_text} **{target_var}** "
                                    f"(avg contribution {avg_contrib:.4f}). Linear correlation with {target_var}: {corr_text}."
                                )
                        else:
                            st.info("No contribution data available for interpretation.")
                        
                        st.markdown("### 💡 Interpretation")
                        if not importance_df.empty:
                            top_feature = importance_df.iloc[0]['Feature']
                            top_pct = importance_df.iloc[0]['Percentage']
                            st.success(f"**Most Important Feature**: {top_feature} ({top_pct:.1f}% of total importance)")
                        else:
                            st.info("Feature importance results unavailable.")
                        
                        if r2 > 0.7:
                            st.info(f"**Strong Model**: R² = {r2:.3f} indicates the selected features explain {r2*100:.1f}% of variance in {target_var}")
                        elif r2 > 0.4:
                            st.info(f"**Moderate Model**: R² = {r2:.3f} indicates the selected features explain {r2*100:.1f}% of variance in {target_var}")
                        else:
                            st.warning(f"**Weak Model**: R² = {r2:.3f} indicates the selected features only explain {r2*100:.1f}% of variance in {target_var}. Consider adding more relevant features.")
                        
                        if len(importance_df) >= 3:
                            st.markdown("**Top 3 Most Important Features:**")
                            for i in range(min(3, len(importance_df))):
                                feat = importance_df.iloc[i]['Feature']
                                pct = importance_df.iloc[i]['Percentage']
                                st.write(f"{i+1}. **{feat}** - {pct:.1f}%")
                
            except ImportError:
                st.error("❌ XGBoost not installed. Install with: `pip install xgboost`")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.code(traceback.format_exc())

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style="text-align: center; padding: 20px; color: var(--text-medium);">
    <p><strong>📊 EDA Insights</strong> • Professional Data Analysis Platform • Version 2.0.0</p>
    <p style="font-size: 0.85em; margin-top: 10px;">
        Multi-file support • Advanced analytics • Interactive visualizations
    </p>
</div>
""", unsafe_allow_html=True)
