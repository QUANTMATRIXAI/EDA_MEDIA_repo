
import io
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Campaign Comparison', layout='wide')

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = BASE_DIR / '01 Raw Data' / 'GA_RegionMapped.xlsx'


@st.cache_data(show_spinner=False)
def load_data_path(path, sheet_name=None):
    ext = Path(path).suffix.lower()
    if ext in ['.xlsx', '.xls']:
        if sheet_name:
            return pd.read_excel(path, sheet_name=sheet_name)
        return pd.read_excel(path)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_data_bytes(file_bytes, filename, sheet_name=None):
    ext = Path(filename).suffix.lower()
    buffer = io.BytesIO(file_bytes)
    if ext in ['.xlsx', '.xls']:
        if sheet_name:
            return pd.read_excel(buffer, sheet_name=sheet_name)
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


def list_sheets_from_upload(file_bytes):
    try:
        excel = pd.ExcelFile(io.BytesIO(file_bytes))
        return excel.sheet_names
    except Exception:
        return []


def list_sheets_from_path(path):
    try:
        excel = pd.ExcelFile(path)
        return excel.sheet_names
    except Exception:
        return []


def find_date_candidates(df):
    keywords = ['date', 'day', 'time', 'dt']
    candidates = [c for c in df.columns if any(k in c.lower() for k in keywords)]
    if candidates:
        return candidates

    sample = df.head(1000)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
            continue
        parsed = pd.to_datetime(sample[col], errors='coerce')
        if parsed.notna().mean() >= 0.6:
            candidates.append(col)

    return list(dict.fromkeys(candidates))


def normalize_date(series):
    return pd.to_datetime(series, errors='coerce').dt.normalize()


def ratio_value(df, numerator, denominator):
    if numerator not in df.columns or denominator not in df.columns:
        return np.nan
    num = df[numerator].sum()
    den = df[denominator].sum()
    if den == 0:
        return np.nan
    return num / den


def compute_ratio_table(df, ratios, periods, masks):
    rows = []
    for metric_name, (num_col, den_col) in ratios.items():
        row = {'Metric': metric_name}
        for label, mask in masks.items():
            for period_label, (start, end) in periods.items():
                period_df = df[(df['date'] >= start) & (df['date'] <= end) & mask]
                row[f'{label} {period_label}'] = ratio_value(period_df, num_col, den_col)
        rows.append(row)

    return pd.DataFrame(rows)


def is_weighted_avg_metric(metric_name):
    name = metric_name.lower()
    if 'session duration' not in name:
        return False
    return 'avg' in name or 'average' in name


def compute_sum_table(df, metrics, periods, masks, divisors=None, session_col='Sessions'):
    rows = []
    for metric_col in metrics:
        row = {'Metric': metric_col}
        for label, mask in masks.items():
            for period_label, (start, end) in periods.items():
                period_df = df[(df['date'] >= start) & (df['date'] <= end) & mask]
                if period_df.empty:
                    total = np.nan
                elif is_weighted_avg_metric(metric_col) and session_col in period_df.columns:
                    weighted_sum = (period_df[metric_col] * period_df[session_col]).sum()
                    session_total = period_df[session_col].sum()
                    total = weighted_sum / session_total if session_total else np.nan
                else:
                    total = period_df[metric_col].sum()
                divisor = 1
                if divisors and label in divisors and divisors[label]:
                    divisor = divisors[label]
                if divisor in [0, None] or pd.isna(divisor):
                    value = np.nan
                else:
                    value = total / divisor
                row[f'{label} {period_label}'] = value
        rows.append(row)

    return pd.DataFrame(rows)


def filter_ratio_defs(ratio_defs, columns):
    filtered = {}
    missing = []
    for name, (num, den) in ratio_defs.items():
        if num in columns and den in columns:
            filtered[name] = (num, den)
        else:
            missing.append(name)
    return filtered, missing


def build_segment_preview(df, segments):
    rows = []
    for name, mask in segments.items():
        seg = df.loc[mask]
        rows.append({
            'Segment': name,
            'Rows': len(seg),
            'Unique dates': seg['date'].nunique(),
            'Date min': seg['date'].min(),
            'Date max': seg['date'].max(),
        })
    return pd.DataFrame(rows)


def build_combined_label(df, cols, sep=' | '):
    def combine_row(row):
        parts = []
        for val in row:
            if pd.isna(val):
                continue
            sval = str(val).strip()
            if not sval or sval.lower() == 'nan':
                continue
            parts.append(sval)
        return sep.join(parts)

    return df[cols].apply(combine_row, axis=1)


def format_indexed_table(table, segments, periods, value_formatter):
    if table.empty:
        return table, table

    formatted = table.copy()
    index_values = pd.DataFrame(np.nan, index=table.index, columns=table.columns)
    for segment in segments:
        pre_col = f'{segment} Pre'
        if pre_col not in formatted.columns:
            continue
        pre_vals = formatted[pre_col]
        for period in periods:
            col = f'{segment} {period}'
            if col not in formatted.columns:
                continue
            values = formatted[col]
            if period == 'Pre':
                idx_vals = np.where(pre_vals.notna() & (pre_vals != 0), 100.0, np.nan)
            else:
                idx_vals = np.where(pre_vals.notna() & (pre_vals != 0), (values / pre_vals) * 100, np.nan)

            index_values[col] = idx_vals
            formatted[col] = [
                format_value(val, idx_val, value_formatter)
                for val, idx_val in zip(values, idx_vals)
            ]

    return formatted, index_values


def build_index_styles(index_values, highlight_periods):
    styles = pd.DataFrame('', index=index_values.index, columns=index_values.columns)
    for period in highlight_periods:
        suffix = f' {period}'
        for col in index_values.columns:
            if not col.endswith(suffix):
                continue
            idx_vals = index_values[col]
            styles[col] = np.where(
                idx_vals.isna(),
                '',
                np.where(
                    idx_vals > 100,
                    'background-color: #d4edda; color: #155724;',
                    'background-color: #f8d7da; color: #721c24;'
                )
            )
    return styles


def format_value(val, idx, value_formatter):
    if pd.isna(val):
        return ''
    val_str = value_formatter(val)
    if pd.isna(idx):
        return f'{val_str} (-)'
    return f'{val_str} ({idx:,.0f})'


def format_ratio(val):
    return f'{val:.4f}'


def format_raw(val):
    if pd.isna(val):
        return ''
    if abs(val - round(val)) < 1e-6:
        return f'{val:,.0f}'
    return f'{val:,.2f}'


def metric_key(name):
    return (
        'metric_'
        + name.lower()
        .replace(' ', '_')
        .replace('/', '_')
        .replace('-', '_')
    )


def clamp_range(start, end, min_date, max_date):
    start = max(start, min_date)
    end = min(end, max_date)
    if end < start:
        end = start
    return start, end


def key(prefix, name):
    return f'{prefix}_{name}'


st.title('Campaign vs Pre-Campaign Comparison')

tabs_main = st.tabs(['Generic dataset', 'Shopify + Meta'])
with tabs_main[0]:
    st.subheader('Controls')
    row1 = st.columns([2.2, 2.2, 1.0, 1.1, 1.1, 1.1])

    with row1[0]:
        upload = st.file_uploader(
            'Upload data file',
            type=['csv', 'xlsx', 'xls'],
            key=key('gen', 'upload')
        )
    with row1[1]:
        if upload is None:
            data_path = st.text_input('Data file path', str(DEFAULT_DATA), key=key('gen', 'path'))
        else:
            data_path = ''
            st.caption('Using uploaded file')

    upload_bytes = None
    upload_name = None
    upload_ext = None
    if upload is not None:
        upload_bytes = upload.getvalue()
        upload_name = upload.name
        upload_ext = Path(upload_name).suffix.lower()

    sheet_name = ''
    if upload_bytes and upload_ext in ['.xlsx', '.xls']:
        sheet_names = list_sheets_from_upload(upload_bytes)
        if sheet_names:
            with row1[2]:
                sheet_name = st.selectbox('Sheet', sheet_names, key=key('gen', 'sheet_select'))
        else:
            with row1[2]:
                sheet_name = st.text_input('Sheet (Excel only)', value='', key=key('gen', 'sheet_text'))
    else:
        with row1[2]:
            sheet_name = st.text_input('Sheet (Excel only)', value='', key=key('gen', 'sheet_text'))

    try:
        if upload_bytes:
            df = load_data_bytes(upload_bytes, upload_name, sheet_name.strip() or None)
        else:
            df = load_data_path(data_path, sheet_name.strip() or None)
    except Exception as exc:
        st.error(f'Failed to load data file: {exc}')
        st.stop()

    if df.empty:
        st.error('Loaded data is empty.')
        st.stop()

    candidate_dates = find_date_candidates(df)
    if not candidate_dates:
        st.error('No date-like columns found. Rename the date column to include date/day or select a different file.')
        st.stop()

    with row1[3]:
        date_col = st.selectbox('Date column', candidate_dates, index=0, key=key('gen', 'date_col'))

    non_numeric_cols = [
        c for c in df.columns
        if c != date_col and not pd.api.types.is_numeric_dtype(df[c])
    ]
    non_numeric_cols = sorted(non_numeric_cols)

    source_options = ['(none)'] + non_numeric_cols
    region_options = ['(none)'] + non_numeric_cols

    source_default = source_options.index('Session source') if 'Session source' in source_options else 0
    region_default = region_options.index('Region_y') if 'Region_y' in region_options else 0

    with row1[4]:
        source_col = st.selectbox('Source column', source_options, index=source_default, key=key('gen', 'source_col'))
    with row1[5]:
        region_col = st.selectbox('Region column', region_options, index=region_default, key=key('gen', 'region_col'))

    working_df = df.copy()
    working_df['date'] = normalize_date(working_df[date_col])
    working_df = working_df[working_df['date'].notna()].copy()

    if working_df.empty:
        st.error('No usable dates found after parsing the date column.')
        st.stop()

    min_date = working_df['date'].min()
    max_date = working_df['date'].max()

    campaign_length = 30
    campaign_start_default = max_date - timedelta(days=campaign_length - 1)
    campaign_end_default = max_date
    pre_end_default = campaign_start_default - timedelta(days=1)
    pre_start_default = pre_end_default - timedelta(days=campaign_length - 1)
    post_start_default = campaign_end_default + timedelta(days=1)
    post_end_default = post_start_default + timedelta(days=campaign_length - 1)

    pre_start_default, pre_end_default = clamp_range(pre_start_default, pre_end_default, min_date, max_date)
    campaign_start_default, campaign_end_default = clamp_range(
        campaign_start_default, campaign_end_default, min_date, max_date
    )
    post_start_default, post_end_default = clamp_range(post_start_default, post_end_default, min_date, max_date)

    row2 = st.columns(6)
    with row2[0]:
        pre_start = st.date_input('Pre start', value=pre_start_default.date(), key=key('gen', 'pre_start'))
    with row2[1]:
        pre_end = st.date_input('Pre end', value=pre_end_default.date(), key=key('gen', 'pre_end'))
    with row2[2]:
        campaign_start = st.date_input(
            'Campaign start', value=campaign_start_default.date(), key=key('gen', 'campaign_start')
        )
    with row2[3]:
        campaign_end = st.date_input(
            'Campaign end', value=campaign_end_default.date(), key=key('gen', 'campaign_end')
        )
    with row2[4]:
        post_start = st.date_input('Post start', value=post_start_default.date(), key=key('gen', 'post_start'))
    with row2[5]:
        post_end = st.date_input('Post end', value=post_end_default.date(), key=key('gen', 'post_end'))

    pre_start = pd.Timestamp(pre_start)
    pre_end = pd.Timestamp(pre_end)
    campaign_start = pd.Timestamp(campaign_start)
    campaign_end = pd.Timestamp(campaign_end)
    post_start = pd.Timestamp(post_start)
    post_end = pd.Timestamp(post_end)

    if pre_end < pre_start or campaign_end < campaign_start or post_end < post_start:
        st.error('Each period must have an end date on or after its start date.')
        st.stop()

    st.caption(f'Data range: {min_date.date()} to {max_date.date()}')

    if pre_start < min_date or pre_end > max_date:
        st.warning('Pre period is outside the available date range.')
    if campaign_start < min_date or campaign_end > max_date:
        st.warning('Campaign period is outside the available date range.')
    if post_start < min_date or post_end > max_date:
        st.warning('Post period is outside the available date range.')

    periods = {
        'Pre': (pre_start, pre_end),
        'Campaign': (campaign_start, campaign_end),
        'Post': (post_start, post_end),
    }

    source_mode = 'None'
    meta_pattern = ''
    include_unknown_source = True
    include_other_source = False
    apply_groups_to_non_meta = True
    include_non_meta_other = True
    source_groups_key = key('gen', 'source_groups')
    if source_groups_key not in st.session_state:
        st.session_state[source_groups_key] = []

    with st.expander('Filters', expanded=True):
        if region_col != '(none)':
            region_values = sorted([v for v in working_df[region_col].dropna().unique()])
            default_focus = ['West'] if 'West' in region_values else region_values[:1]
            focus_regions = st.multiselect(
                'Focus region values',
                region_values,
                default=default_focus,
                key=key('gen', 'focus_regions')
            )
            include_unknown_region = st.checkbox(
                'Include unknown region in All Other',
                value=True,
                key=key('gen', 'include_unknown_region')
            )
        else:
            focus_regions = []
            include_unknown_region = True
            st.info('Select a region column to enable region filtering.')

        if source_col != '(none)':
            st.markdown('**Source segmentation**')
            source_mode = st.radio(
                'Source segmentation mode',
                ['Meta regex', 'Custom groups', 'Meta + Custom groups'],
                horizontal=True,
                key=key('gen', 'source_mode')
            )

            if source_mode in ['Meta regex', 'Meta + Custom groups']:
                default_pattern = r'fb|fbig|facebook|meta|insta'
                meta_pattern = st.text_input(
                    'Meta source pattern (regex)',
                    value=default_pattern,
                    key=key('gen', 'meta_pattern')
                )
                include_unknown_source = st.checkbox(
                    'Include unknown source in Non-Meta/Other',
                    value=True,
                    key=key('gen', 'include_unknown_source')
                )

            if source_mode in ['Custom groups', 'Meta + Custom groups']:
                source_values = sorted(
                    working_df[source_col].dropna().astype('string').unique()
                )
                group_name = st.text_input('Group name', key=key('gen', 'group_name'))
                group_values = st.multiselect(
                    'Group values',
                    source_values,
                    key=key('gen', 'group_values')
                )
                add_group = st.button('Add group', key=key('gen', 'add_group'))
                if add_group and group_name.strip() and group_values:
                    new_name = group_name.strip()
                    st.session_state[source_groups_key] = [
                        g for g in st.session_state[source_groups_key]
                        if g['name'] != new_name
                    ] + [{'name': new_name, 'values': group_values}]
                    st.rerun()

                if st.session_state[source_groups_key]:
                    group_names = [g['name'] for g in st.session_state[source_groups_key]]
                    st.caption('Current groups:')
                    for group in st.session_state[source_groups_key]:
                        st.write(f"- {group['name']}: {', '.join(str(v) for v in group['values'][:6])}")
                    remove_name = st.selectbox(
                        'Remove group',
                        ['(none)'] + group_names,
                        key=key('gen', 'remove_group')
                    )
                    remove_group = st.button('Remove', key=key('gen', 'remove_group_btn'))
                    if remove_group and remove_name != '(none)':
                        st.session_state[source_groups_key] = [
                            g for g in st.session_state[source_groups_key]
                            if g['name'] != remove_name
                        ]
                        st.rerun()
                else:
                    st.info('Add at least one group to enable custom source segmentation.')

                if source_mode == 'Custom groups':
                    include_other_source = st.checkbox(
                        'Include Other sources',
                        value=True,
                        key=key('gen', 'include_other_source')
                    )
                    include_unknown_source = st.checkbox(
                        'Include unknown in Other',
                        value=True,
                        key=key('gen', 'include_unknown_source_custom')
                    )
                else:
                    apply_groups_to_non_meta = st.checkbox(
                        'Apply custom groups to Non-Meta only',
                        value=True,
                        key=key('gen', 'apply_groups_non_meta')
                    )
                    include_non_meta_other = st.checkbox(
                        'Include Non-Meta Other',
                        value=True,
                        key=key('gen', 'include_non_meta_other')
                    )
        else:
            include_unknown_source = True
            st.warning('Select a source column to enable source segmentation.')

    numeric_cols = [c for c in working_df.select_dtypes(include='number').columns if c != 'date']

    ratio_candidates = [
        ('Engaged Sessions / Sessions', 'Engaged sessions', 'Sessions'),
        ('New Users / Total Users', 'New users', 'Total users'),
        ('Items Viewed / Total Users', 'Items viewed', 'Total users'),
        ('Add to Carts / Sessions', 'Add to carts', 'Sessions'),
        ('Total Purchasers / Total Users', 'Total purchasers', 'Total users'),
    ]

    ratio_defs = {
        name: (num, den)
        for name, num, den in ratio_candidates
        if num in working_df.columns and den in working_df.columns
    }

    custom_ratios_key = key('gen', 'custom_ratios')
    selected_metrics_key = key('gen', 'selected_metrics')
    metric_mode_key = key('gen', 'metric_mode')
    metric_mode_prev_key = key('gen', 'metric_mode_prev')

    if custom_ratios_key not in st.session_state:
        st.session_state[custom_ratios_key] = {}

    ratio_defs.update(st.session_state[custom_ratios_key])
    ratio_defs, missing_ratios = filter_ratio_defs(ratio_defs, working_df.columns)
    if missing_ratios:
        st.warning(
            'Some ratios were hidden because columns are missing: '
            + ', '.join(missing_ratios)
        )

    if selected_metrics_key not in st.session_state:
        st.session_state[selected_metrics_key] = list(ratio_defs.keys())
    else:
        st.session_state[selected_metrics_key] = [
            m for m in st.session_state[selected_metrics_key] if m in ratio_defs
        ]

    st.subheader('Metrics')
    metric_mode = st.radio(
        'Metrics selection mode',
        ['Multiselect', 'Checklist'],
        horizontal=True,
        key=metric_mode_key,
    )

    metric_keys = {name: metric_key(name) for name in ratio_defs}

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    select_all = btn_col1.button('Select all', key=key('gen', 'select_all'))
    clear_all = btn_col2.button('Clear', key=key('gen', 'clear_all'))

    if select_all:
        st.session_state[selected_metrics_key] = list(ratio_defs.keys())
        for name, key_name in metric_keys.items():
            st.session_state[f"{key('gen', 'metric')}_{key_name}"] = True
    if clear_all:
        st.session_state[selected_metrics_key] = []
        for name, key_name in metric_keys.items():
            st.session_state[f"{key('gen', 'metric')}_{key_name}"] = False

    if metric_mode == 'Checklist':
        if st.session_state.get(metric_mode_prev_key) != 'Checklist':
            for name, key_name in metric_keys.items():
                st.session_state[f"{key('gen', 'metric')}_{key_name}"] = (
                    name in st.session_state[selected_metrics_key]
                )
        for name, key_name in metric_keys.items():
            widget_key = f"{key('gen', 'metric')}_{key_name}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = name in st.session_state[selected_metrics_key]
        check_cols = st.columns(3)
        for idx, name in enumerate(ratio_defs.keys()):
            widget_key = f"{key('gen', 'metric')}_{metric_keys[name]}"
            with check_cols[idx % 3]:
                st.checkbox(name, key=widget_key)
        selected_metrics = [
            name for name in ratio_defs.keys()
            if st.session_state.get(f"{key('gen', 'metric')}_{metric_keys[name]}")
        ]
        st.session_state[selected_metrics_key] = selected_metrics
    else:
        if st.session_state.get(metric_mode_prev_key) == 'Checklist':
            selected_metrics = [
                name for name in ratio_defs.keys()
                if st.session_state.get(f"{key('gen', 'metric')}_{metric_keys[name]}")
            ]
            st.session_state[selected_metrics_key] = selected_metrics
        st.multiselect(
            'Ratios to display',
            list(ratio_defs.keys()),
            key=selected_metrics_key,
        )
        selected_metrics = st.session_state[selected_metrics_key]

    st.session_state[metric_mode_prev_key] = metric_mode

    with st.expander('Add custom ratio'):
        custom_name = st.text_input('Custom ratio name', value='', key=key('gen', 'custom_ratio_name'))
        custom_num = st.selectbox('Numerator column', numeric_cols, key=key('gen', 'custom_ratio_num'))
        custom_den = st.selectbox(
            'Denominator column',
            numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0,
            key=key('gen', 'custom_ratio_den'),
        )
        add_custom = st.button('Add ratio', key=key('gen', 'add_ratio'))
        if add_custom and custom_name.strip():
            name = custom_name.strip()
            st.session_state[custom_ratios_key][name] = (custom_num, custom_den)
            ratio_defs[name] = (custom_num, custom_den)
            if name not in st.session_state[selected_metrics_key]:
                st.session_state[selected_metrics_key].append(name)
            st.rerun()

    ratios = {k: v for k, v in ratio_defs.items() if k in st.session_state[selected_metrics_key]}

    raw_metrics = st.multiselect(
        'Raw metrics (period total / region count)',
        numeric_cols,
        default=[],
        help='These are period totals divided by the number of focus/other regions.',
        key=key('gen', 'raw_metrics'),
    )
    if region_col != '(none)' and focus_regions:
        region_series = working_df[region_col]
        focus_mask = region_series.isin(focus_regions)
        if include_unknown_region:
            other_mask = ~focus_mask
        else:
            other_mask = (~focus_mask) & region_series.notna()
        show_other = True
        normalized_regions = [str(v).strip().replace(' ', '_') for v in focus_regions]
        focus_label = f"{region_col}: {'_'.join(normalized_regions)}"
        other_label = 'All Other Regions'
    else:
        focus_mask = pd.Series(True, index=working_df.index)
        other_mask = pd.Series(False, index=working_df.index)
        show_other = False
        focus_label = 'All Data'
        other_label = 'All Other Regions'

    source_segments = {'All Data': pd.Series(True, index=working_df.index)}
    segment_note = '(All Data)'
    if source_col != '(none)':
        source_series = working_df[source_col].astype('string').str.strip()
        source_norm = source_series.str.lower()

        if source_mode in ['Meta regex', 'Meta + Custom groups']:
            try:
                pattern = re.compile(meta_pattern, flags=re.IGNORECASE)
            except re.error as exc:
                st.error(f'Invalid regex pattern: {exc}')
                st.stop()
            meta_mask = source_norm.str.contains(pattern, na=False)
        else:
            meta_mask = pd.Series(False, index=working_df.index)

        if source_mode == 'Meta regex':
            if include_unknown_source:
                non_meta_mask = ~meta_mask
            else:
                non_meta_mask = (~meta_mask) & source_series.notna()
            source_segments = {'Meta': meta_mask, 'Non-Meta': non_meta_mask}
            segment_note = '(Meta vs Non-Meta)'
        elif source_mode == 'Custom groups':
            groups = st.session_state[source_groups_key]
            if groups:
                used_mask = pd.Series(False, index=working_df.index)
                source_segments = {}
                for group in groups:
                    values = [str(v).lower() for v in group['values']]
                    mask = source_norm.isin(values)
                    source_segments[group['name']] = mask
                    used_mask = used_mask | mask
                if include_other_source:
                    other_mask = ~used_mask
                    if not include_unknown_source:
                        other_mask = other_mask & source_series.notna()
                    source_segments['Other'] = other_mask
                segment_note = '(Custom source groups)'
        else:
            groups = st.session_state[source_groups_key]
            if groups:
                if include_unknown_source:
                    non_meta_mask = ~meta_mask
                else:
                    non_meta_mask = (~meta_mask) & source_series.notna()
                used_mask = pd.Series(False, index=working_df.index)
                source_segments = {'Meta': meta_mask, 'Non-Meta': non_meta_mask}
                for group in groups:
                    values = [str(v).lower() for v in group['values']]
                    group_mask = source_norm.isin(values)
                    if apply_groups_to_non_meta:
                        group_mask = group_mask & (~meta_mask)
                    group_name = group['name']
                    if group_name in source_segments:
                        group_name = f'{group_name} (group)'
                    source_segments[group_name] = group_mask
                    used_mask = used_mask | group_mask

                if include_non_meta_other:
                    other_mask = non_meta_mask & ~used_mask
                    source_segments['Non-Meta Other'] = other_mask
                segment_note = '(Meta + custom groups)'
            else:
                if include_unknown_source:
                    non_meta_mask = ~meta_mask
                else:
                    non_meta_mask = (~meta_mask) & source_series.notna()
                source_segments = {'Meta': meta_mask, 'Non-Meta': non_meta_mask}
                segment_note = '(Meta vs Non-Meta)'

    focus_segments = {label: focus_mask & mask for label, mask in source_segments.items()}
    other_segments = {label: other_mask & mask for label, mask in source_segments.items()}

    focus_region_count = 1
    other_region_count = np.nan
    if region_col != '(none)':
        if focus_regions:
            focus_region_count = len(focus_regions)
        else:
            focus_region_count = max(1, region_series[focus_mask].dropna().nunique())
        if show_other:
            other_region_count = region_series[other_mask].dropna().nunique()
            if other_region_count == 0:
                other_region_count = np.nan

    with st.expander('Data preview (post-filters)', expanded=False):
        st.write(f'Rows: {len(working_df):,}')
        st.write(f'Date range: {min_date.date()} to {max_date.date()}')
        st.write(f'Date column: {date_col}')
        st.write(f'Source column: {source_col}')
        st.write(f'Region column: {region_col}')
        focus_label_preview = '_'.join(
            [str(v).strip().replace(' ', '_') for v in focus_regions]
        ) if focus_regions else 'All'
        st.write(f'Focus regions: {focus_label_preview}')
        st.write(f'Source mode: {source_mode}')
        if source_mode in ['Meta regex', 'Meta + Custom groups']:
            st.write(f'Meta pattern: {meta_pattern}')
        if source_mode in ['Custom groups', 'Meta + Custom groups']:
            group_list = [g["name"] for g in st.session_state[source_groups_key]]
            st.write(f'Source groups: {", ".join(group_list) if group_list else "None"}')
        st.write(f'Focus region divisor: {focus_region_count}')
        if show_other:
            st.write(
                f'Other region divisor: '
                f'{other_region_count if not pd.isna(other_region_count) else "N/A"}'
            )
        st.write('Raw metric rule: totals / region count; Average session duration uses sessions-weighted average.')
        st.write('Segment counts')
        st.dataframe(build_segment_preview(working_df, focus_segments), use_container_width=True)
        if show_other:
            st.dataframe(build_segment_preview(working_df, other_segments), use_container_width=True)
        st.write('Sample rows (focus)')
        st.dataframe(working_df.loc[focus_mask].head(20), use_container_width=True)
        if show_other:
            st.write('Sample rows (other)')
            st.dataframe(working_df.loc[other_mask].head(20), use_container_width=True)

    ratio_focus_table = compute_ratio_table(working_df, ratios, periods, focus_segments)
    focus_divisors = {label: focus_region_count for label in focus_segments}
    raw_focus_table = compute_sum_table(
        working_df, raw_metrics, periods, focus_segments, divisors=focus_divisors
    )

    ratio_other_table = compute_ratio_table(working_df, ratios, periods, other_segments)
    other_divisors = {label: other_region_count for label in other_segments}
    raw_other_table = compute_sum_table(
        working_df, raw_metrics, periods, other_segments, divisors=other_divisors
    )

    segment_labels = list(focus_segments.keys())
    period_labels = list(periods.keys())
    highlight_periods = ['Campaign', 'Post']

    ratio_focus_table_display, ratio_focus_index = format_indexed_table(
        ratio_focus_table, segment_labels, period_labels, format_ratio
    )
    ratio_other_table_display, ratio_other_index = format_indexed_table(
        ratio_other_table, segment_labels, period_labels, format_ratio
    )

    raw_focus_table_display, raw_focus_index = format_indexed_table(
        raw_focus_table, segment_labels, period_labels, format_raw
    )
    raw_other_table_display, raw_other_index = format_indexed_table(
        raw_other_table, segment_labels, period_labels, format_raw
    )

    tabs = st.tabs(['Ratios', 'Raw Metrics'])

    with tabs[0]:
        st.subheader(f'{focus_label} {segment_note}')
        if ratio_focus_table_display.empty:
            st.info('No ratio metrics selected or no data for the selected filters.')
        else:
            ratio_focus_styles = build_index_styles(ratio_focus_index, highlight_periods)
            st.dataframe(
                ratio_focus_table_display.style.apply(lambda _: ratio_focus_styles, axis=None),
                use_container_width=True
            )

        if show_other:
            st.subheader(f'{other_label} {segment_note}')
            if ratio_other_table_display.empty:
                st.info('No ratio metrics selected or no data for the selected filters.')
            else:
                ratio_other_styles = build_index_styles(ratio_other_index, highlight_periods)
                st.dataframe(
                    ratio_other_table_display.style.apply(lambda _: ratio_other_styles, axis=None),
                    use_container_width=True
                )

    with tabs[1]:
        st.subheader(f'{focus_label} {segment_note}')
        st.caption(f'Raw metric divisor: {focus_region_count}')
        if raw_focus_table_display.empty:
            st.info('No raw metrics selected or no data for the selected filters.')
        else:
            raw_focus_styles = build_index_styles(raw_focus_index, highlight_periods)
            st.dataframe(
                raw_focus_table_display.style.apply(lambda _: raw_focus_styles, axis=None),
                use_container_width=True
            )

    if show_other:
        st.subheader(f'{other_label} {segment_note}')
        st.caption(
            f'Raw metric divisor: '
            f'{other_region_count if not pd.isna(other_region_count) else "N/A"}'
        )
        if raw_other_table_display.empty:
            st.info('No raw metrics selected or no data for the selected filters.')
        else:
            raw_other_styles = build_index_styles(raw_other_index, highlight_periods)
            st.dataframe(
                    raw_other_table_display.style.apply(lambda _: raw_other_styles, axis=None),
                    use_container_width=True
                )

st.caption(
    'Notes: Each cell is value (index). Index uses Pre = 100 for each segment. '
    'Raw metrics are period totals divided by region count. '
    'Average session duration uses sessions-weighted average before the divisor.'
)
with tabs_main[1]:
    st.subheader('Shopify + Meta Controls')
    row1 = st.columns([2.0, 2.0, 1.0, 2.0, 2.0, 1.0])

    with row1[0]:
        shopify_upload = st.file_uploader(
            'Upload Shopify file',
            type=['csv', 'xlsx', 'xls'],
            key=key('sm', 'shopify_upload')
        )
    shopify_path = ''

    with row1[3]:
        meta_upload = st.file_uploader(
            'Upload Meta file',
            type=['csv', 'xlsx', 'xls'],
            key=key('sm', 'meta_upload')
        )
    meta_path = ''

    shopify_bytes = None
    shopify_name = None
    shopify_ext = None
    if shopify_upload is not None:
        shopify_bytes = shopify_upload.getvalue()
        shopify_name = shopify_upload.name
        shopify_ext = Path(shopify_name).suffix.lower()

    meta_bytes = None
    meta_name = None
    meta_ext = None
    if meta_upload is not None:
        meta_bytes = meta_upload.getvalue()
        meta_name = meta_upload.name
        meta_ext = Path(meta_name).suffix.lower()

    shopify_sheet = ''
    if shopify_bytes and shopify_ext in ['.xlsx', '.xls']:
        sheet_names = list_sheets_from_upload(shopify_bytes)
        if sheet_names:
            with row1[2]:
                shopify_sheet = st.selectbox('Shopify sheet', sheet_names, key=key('sm', 'shopify_sheet'))
        else:
            with row1[2]:
                shopify_sheet = st.text_input('Shopify sheet', value='', key=key('sm', 'shopify_sheet_text'))
    else:
        with row1[2]:
            shopify_sheet = st.text_input('Shopify sheet', value='', key=key('sm', 'shopify_sheet_text'))

    meta_sheet = ''
    if meta_bytes and meta_ext in ['.xlsx', '.xls']:
        sheet_names = list_sheets_from_upload(meta_bytes)
        if sheet_names:
            with row1[5]:
                meta_sheet = st.selectbox('Meta sheet', sheet_names, key=key('sm', 'meta_sheet'))
        else:
            with row1[5]:
                meta_sheet = st.text_input('Meta sheet', value='', key=key('sm', 'meta_sheet_text'))
    else:
        with row1[5]:
            meta_sheet = st.text_input('Meta sheet', value='', key=key('sm', 'meta_sheet_text'))

    if not shopify_bytes or not meta_bytes:
        st.info('Upload both Shopify and Meta files to continue.')
        st.stop()

    try:
        shopify_df = load_data_bytes(shopify_bytes, shopify_name, shopify_sheet.strip() or None)
        meta_df = load_data_bytes(meta_bytes, meta_name, meta_sheet.strip() or None)
    except Exception as exc:
        st.error(f'Failed to load Shopify/Meta files: {exc}')
        st.stop()

    if shopify_df.empty or meta_df.empty:
        st.error('Shopify or Meta dataset is empty.')
        st.stop()

    shopify_date_candidates = find_date_candidates(shopify_df)
    meta_date_candidates = find_date_candidates(meta_df)

    if not shopify_date_candidates or not meta_date_candidates:
        st.error('Date column not found in Shopify or Meta dataset.')
        st.stop()

    pivot_groups_key = key('sm', 'pivot_groups')
    if pivot_groups_key not in st.session_state:
        st.session_state[pivot_groups_key] = []

    pivot_mode = 'Use raw categories'
    include_other_pivot = True
    include_unknown_pivot = True

    with st.expander('Shopify + Meta settings', expanded=True):
        shopify_date_col = st.selectbox(
            'Shopify date column',
            shopify_date_candidates,
            index=0,
            key=key('sm', 'shopify_date_col')
        )
        meta_date_col = st.selectbox(
            'Meta date column',
            meta_date_candidates,
            index=0,
            key=key('sm', 'meta_date_col')
        )

        shopify_non_numeric = [
            c for c in shopify_df.columns
            if c != shopify_date_col and not pd.api.types.is_numeric_dtype(shopify_df[c])
        ]
        default_cat = None
        for c in shopify_non_numeric:
            if 'new' in c.lower() and 'return' in c.lower():
                default_cat = c
                break
        if not default_cat and shopify_non_numeric:
            default_cat = shopify_non_numeric[0]

        default_pivot_cols = [default_cat] if default_cat else []
        pivot_cols = st.multiselect(
            'Shopify pivot columns',
            shopify_non_numeric,
            default=default_pivot_cols,
            key=key('sm', 'shopify_pivot_cols')
        )

        shopify_numeric = [
            c for c in shopify_df.select_dtypes(include='number').columns
            if c != shopify_date_col
        ]
        default_metrics = [c for c in ['Net sales', 'Orders'] if c in shopify_numeric]
        if not default_metrics:
            default_metrics = shopify_numeric[:2]
        shopify_metrics = st.multiselect(
            'Shopify metrics to pivot',
            shopify_numeric,
            default=default_metrics,
            key=key('sm', 'shopify_metrics')
        )

        pivot_mode = st.radio(
            'Pivot categories',
            ['Use raw categories', 'Consolidate categories'],
            horizontal=True,
            key=key('sm', 'pivot_mode')
        )

        if pivot_mode == 'Consolidate categories':
            if not pivot_cols:
                st.error('Select at least one pivot column.')
                st.stop()
            temp_labels = build_combined_label(shopify_df, pivot_cols)
            category_values = sorted(temp_labels.dropna().unique())
            group_name = st.text_input('Group name', key=key('sm', 'pivot_group_name'))
            group_values = st.multiselect(
                'Group values',
                category_values,
                key=key('sm', 'pivot_group_values')
            )
            add_group = st.button('Add pivot group', key=key('sm', 'pivot_add_group'))
            if add_group and group_name.strip() and group_values:
                new_name = group_name.strip()
                st.session_state[pivot_groups_key] = [
                    g for g in st.session_state[pivot_groups_key]
                    if g['name'] != new_name
                ] + [{'name': new_name, 'values': group_values}]
                st.rerun()

            if st.session_state[pivot_groups_key]:
                group_names = [g['name'] for g in st.session_state[pivot_groups_key]]
                st.caption('Current pivot groups:')
                for group in st.session_state[pivot_groups_key]:
                    st.write(f"- {group['name']}: {', '.join(str(v) for v in group['values'][:6])}")
                remove_name = st.selectbox(
                    'Remove pivot group',
                    ['(none)'] + group_names,
                    key=key('sm', 'pivot_remove_group')
                )
                remove_group = st.button('Remove', key=key('sm', 'pivot_remove_group_btn'))
                if remove_group and remove_name != '(none)':
                    st.session_state[pivot_groups_key] = [
                        g for g in st.session_state[pivot_groups_key]
                        if g['name'] != remove_name
                    ]
                    st.rerun()
            else:
                st.info('Add at least one group to consolidate categories.')

            include_other_pivot = st.checkbox(
                'Include Other categories',
                value=True,
                key=key('sm', 'pivot_include_other')
            )
            include_unknown_pivot = st.checkbox(
                'Include unknown in Other',
                value=True,
                key=key('sm', 'pivot_include_unknown')
            )

    shopify_df['date'] = normalize_date(shopify_df[shopify_date_col])
    meta_df['date'] = normalize_date(meta_df[meta_date_col])

    shopify_df = shopify_df[shopify_df['date'].notna()].copy()
    meta_df = meta_df[meta_df['date'].notna()].copy()

    if shopify_df.empty or meta_df.empty:
        st.error('No usable dates found in Shopify or Meta dataset.')
        st.stop()

    shop_min, shop_max = shopify_df['date'].min(), shopify_df['date'].max()
    meta_min, meta_max = meta_df['date'].min(), meta_df['date'].max()

    overlap_start = max(shop_min, meta_min)
    overlap_end = min(shop_max, meta_max)
    if overlap_end < overlap_start:
        st.error('No overlapping dates between Shopify and Meta datasets.')
        st.stop()

    shopify_df = shopify_df[(shopify_df['date'] >= overlap_start) & (shopify_df['date'] <= overlap_end)]
    meta_df = meta_df[(meta_df['date'] >= overlap_start) & (meta_df['date'] <= overlap_end)]

    if not pivot_cols:
        st.error('Select at least one pivot column.')
        st.stop()

    for col in pivot_cols:
        shopify_df[col] = shopify_df[col].astype('string').str.strip()

    shopify_df['pivot_base'] = build_combined_label(shopify_df, pivot_cols)
    shopify_df = shopify_df[shopify_df['pivot_base'].notna()]
    shopify_df = shopify_df[shopify_df['pivot_base'].str.len() > 0]

    pivot_col = 'pivot_base'
    if pivot_mode == 'Consolidate categories':
        groups = st.session_state[pivot_groups_key]
        if groups:
            cat_series = shopify_df[pivot_col].astype('string')
            cat_norm = cat_series.str.lower()
            group_map = {}
            for group in groups:
                for value in group['values']:
                    group_map[str(value).lower()] = group['name']
            group_series = cat_norm.map(group_map)
            if include_other_pivot:
                if include_unknown_pivot:
                    group_series = group_series.fillna('Other')
                else:
                    group_series = group_series.where(cat_series.notna())
                    group_series = group_series.fillna('Other')
            else:
                group_series = group_series.where(group_series.notna())
            shopify_df = shopify_df[group_series.notna()].copy()
            shopify_df['pivot_group'] = group_series
            pivot_col = 'pivot_group'

    shopify_grouped = (
        shopify_df.groupby(['date', pivot_col], as_index=False)[shopify_metrics]
        .sum()
    )
    shopify_pivot = shopify_grouped.pivot(index='date', columns=pivot_col, values=shopify_metrics)
    shopify_pivot.columns = [
        f"shopify_{metric.lower().replace(' ', '_')}_{str(cat).lower().replace(' ', '_')}"
        for metric, cat in shopify_pivot.columns
    ]
    shopify_pivot = shopify_pivot.reset_index().fillna(0)

    meta_numeric = [
        c for c in meta_df.select_dtypes(include='number').columns
        if c != meta_date_col
    ]
    meta_agg = meta_df.groupby('date', as_index=False)[meta_numeric].sum()
    meta_agg = meta_agg.rename(columns={c: f'meta_{c}' for c in meta_agg.columns if c != 'date'})

    merged = shopify_pivot.merge(meta_agg, on='date', how='inner')

    if merged.empty:
        st.error('Merged Shopify + Meta dataset is empty after pivot/merge.')
        st.stop()

    with st.expander('Data preview', expanded=False):
        st.write(f'Shopify rows: {len(shopify_df):,}')
        st.write(f'Meta rows: {len(meta_df):,}')
        st.write(f'Merged rows: {len(merged):,}')
        st.dataframe(shopify_df.head(20), use_container_width=True)
        st.dataframe(shopify_pivot.head(20), use_container_width=True)
        st.dataframe(meta_agg.head(20), use_container_width=True)
        st.dataframe(merged.head(20), use_container_width=True)
    min_date = merged['date'].min()
    max_date = merged['date'].max()

    campaign_length = 30
    campaign_start_default = max_date - timedelta(days=campaign_length - 1)
    campaign_end_default = max_date
    pre_end_default = campaign_start_default - timedelta(days=1)
    pre_start_default = pre_end_default - timedelta(days=campaign_length - 1)
    post_start_default = campaign_end_default + timedelta(days=1)
    post_end_default = post_start_default + timedelta(days=campaign_length - 1)

    pre_start_default, pre_end_default = clamp_range(pre_start_default, pre_end_default, min_date, max_date)
    campaign_start_default, campaign_end_default = clamp_range(
        campaign_start_default, campaign_end_default, min_date, max_date
    )
    post_start_default, post_end_default = clamp_range(post_start_default, post_end_default, min_date, max_date)

    row2 = st.columns(6)
    with row2[0]:
        pre_start = st.date_input('Pre start', value=pre_start_default.date(), key=key('sm', 'pre_start'))
    with row2[1]:
        pre_end = st.date_input('Pre end', value=pre_end_default.date(), key=key('sm', 'pre_end'))
    with row2[2]:
        campaign_start = st.date_input(
            'Campaign start', value=campaign_start_default.date(), key=key('sm', 'campaign_start')
        )
    with row2[3]:
        campaign_end = st.date_input(
            'Campaign end', value=campaign_end_default.date(), key=key('sm', 'campaign_end')
        )
    with row2[4]:
        post_start = st.date_input('Post start', value=post_start_default.date(), key=key('sm', 'post_start'))
    with row2[5]:
        post_end = st.date_input('Post end', value=post_end_default.date(), key=key('sm', 'post_end'))

    pre_start = pd.Timestamp(pre_start)
    pre_end = pd.Timestamp(pre_end)
    campaign_start = pd.Timestamp(campaign_start)
    campaign_end = pd.Timestamp(campaign_end)
    post_start = pd.Timestamp(post_start)
    post_end = pd.Timestamp(post_end)

    if pre_end < pre_start or campaign_end < campaign_start or post_end < post_start:
        st.error('Each period must have an end date on or after its start date.')
        st.stop()

    st.caption(f'Overlap range: {min_date.date()} to {max_date.date()}')

    periods = {
        'Pre': (pre_start, pre_end),
        'Campaign': (campaign_start, campaign_end),
        'Post': (post_start, post_end),
    }

    numeric_cols = [c for c in merged.select_dtypes(include='number').columns if c != 'date']

    ratio_defs = {}
    if 'meta_Amount spent (USD)' in merged.columns and 'meta_Impressions' in merged.columns:
        ratio_defs['Meta Spend / Impressions'] = ('meta_Amount spent (USD)', 'meta_Impressions')

    custom_ratios_key = key('sm', 'custom_ratios')
    selected_metrics_key = key('sm', 'selected_metrics')
    metric_mode_key = key('sm', 'metric_mode')
    metric_mode_prev_key = key('sm', 'metric_mode_prev')

    if custom_ratios_key not in st.session_state:
        st.session_state[custom_ratios_key] = {}

    ratio_defs.update(st.session_state[custom_ratios_key])
    ratio_defs, missing_ratios = filter_ratio_defs(ratio_defs, merged.columns)
    if missing_ratios:
        st.warning(
            'Some ratios were hidden because columns are missing: '
            + ', '.join(missing_ratios)
        )

    if selected_metrics_key not in st.session_state:
        st.session_state[selected_metrics_key] = list(ratio_defs.keys())
    else:
        st.session_state[selected_metrics_key] = [
            m for m in st.session_state[selected_metrics_key] if m in ratio_defs
        ]

    st.subheader('Metrics')
    metric_mode = st.radio(
        'Metrics selection mode',
        ['Multiselect', 'Checklist'],
        horizontal=True,
        key=metric_mode_key,
    )

    metric_keys = {name: metric_key(name) for name in ratio_defs}

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    select_all = btn_col1.button('Select all', key=key('sm', 'select_all'))
    clear_all = btn_col2.button('Clear', key=key('sm', 'clear_all'))

    if select_all:
        st.session_state[selected_metrics_key] = list(ratio_defs.keys())
        for name, key_name in metric_keys.items():
            st.session_state[f"{key('sm', 'metric')}_{key_name}"] = True
    if clear_all:
        st.session_state[selected_metrics_key] = []
        for name, key_name in metric_keys.items():
            st.session_state[f"{key('sm', 'metric')}_{key_name}"] = False

    if metric_mode == 'Checklist':
        if st.session_state.get(metric_mode_prev_key) != 'Checklist':
            for name, key_name in metric_keys.items():
                st.session_state[f"{key('sm', 'metric')}_{key_name}"] = (
                    name in st.session_state[selected_metrics_key]
                )
        for name, key_name in metric_keys.items():
            widget_key = f"{key('sm', 'metric')}_{key_name}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = name in st.session_state[selected_metrics_key]
        check_cols = st.columns(3)
        for idx, name in enumerate(ratio_defs.keys()):
            widget_key = f"{key('sm', 'metric')}_{metric_keys[name]}"
            with check_cols[idx % 3]:
                st.checkbox(name, key=widget_key)
        selected_metrics = [
            name for name in ratio_defs.keys()
            if st.session_state.get(f"{key('sm', 'metric')}_{metric_keys[name]}")
        ]
        st.session_state[selected_metrics_key] = selected_metrics
    else:
        if st.session_state.get(metric_mode_prev_key) == 'Checklist':
            selected_metrics = [
                name for name in ratio_defs.keys()
                if st.session_state.get(f"{key('sm', 'metric')}_{metric_keys[name]}")
            ]
            st.session_state[selected_metrics_key] = selected_metrics
        st.multiselect(
            'Ratios to display',
            list(ratio_defs.keys()),
            key=selected_metrics_key,
        )
        selected_metrics = st.session_state[selected_metrics_key]

    st.session_state[metric_mode_prev_key] = metric_mode

    with st.expander('Add custom ratio'):
        custom_name = st.text_input('Custom ratio name', value='', key=key('sm', 'custom_ratio_name'))
        custom_num = st.selectbox('Numerator column', numeric_cols, key=key('sm', 'custom_ratio_num'))
        custom_den = st.selectbox(
            'Denominator column',
            numeric_cols,
            index=1 if len(numeric_cols) > 1 else 0,
            key=key('sm', 'custom_ratio_den'),
        )
        add_custom = st.button('Add ratio', key=key('sm', 'add_ratio'))
        if add_custom and custom_name.strip():
            name = custom_name.strip()
            st.session_state[custom_ratios_key][name] = (custom_num, custom_den)
            ratio_defs[name] = (custom_num, custom_den)
            if name not in st.session_state[selected_metrics_key]:
                st.session_state[selected_metrics_key].append(name)
            st.rerun()

    ratios = {k: v for k, v in ratio_defs.items() if k in st.session_state[selected_metrics_key]}

    raw_metrics = st.multiselect(
        'Raw metrics (period totals)',
        numeric_cols,
        default=[],
        help='These are summed over the selected period.',
        key=key('sm', 'raw_metrics'),
    )

    segment_labels = ['All Data']
    segments = {'All Data': pd.Series(True, index=merged.index)}

    ratio_table = compute_ratio_table(merged, ratios, periods, segments)
    raw_table = compute_sum_table(merged, raw_metrics, periods, segments, divisors=None)

    period_labels = list(periods.keys())
    highlight_periods = ['Campaign', 'Post']

    ratio_display, ratio_index = format_indexed_table(ratio_table, segment_labels, period_labels, format_ratio)
    raw_display, raw_index = format_indexed_table(raw_table, segment_labels, period_labels, format_raw)

    tabs = st.tabs(['Ratios', 'Raw Metrics'])

    with tabs[0]:
        if ratio_display.empty:
            st.info('No ratio metrics selected or no data for the selected filters.')
        else:
            ratio_styles = build_index_styles(ratio_index, highlight_periods)
            st.dataframe(
                ratio_display.style.apply(lambda _: ratio_styles, axis=None),
                use_container_width=True
            )

    with tabs[1]:
        if raw_display.empty:
            st.info('No raw metrics selected or no data for the selected filters.')
        else:
            raw_styles = build_index_styles(raw_index, highlight_periods)
            st.dataframe(
                raw_display.style.apply(lambda _: raw_styles, axis=None),
                use_container_width=True
            )

st.caption('Notes: Each cell is value (index). Index uses Pre = 100 for each segment.')
