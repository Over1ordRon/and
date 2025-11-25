# analysis/views.py
# Optimized version that works with your existing backend.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pickle
import hashlib

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache

from .models import DataFile
from .forms import FileUploadForm
from .backend import DataAnalysisBackend  # Using your existing backend


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_cache_key(prefix, *args):
    """Generate cache key from prefix and arguments"""
    key_str = f"{prefix}:{'_'.join(map(str, args))}"
    return hashlib.md5(key_str.encode()).hexdigest()


def store_dataframe_efficiently(df, file_id):
    """Store DataFrame using pickle in cache (more efficient than JSON)"""
    cache_key = f'dataframe_{file_id}'
    # Pickle is much faster and more memory efficient than JSON
    df_pickle = pickle.dumps(df)
    # Cache for 1 hour (3600 seconds)
    cache.set(cache_key, df_pickle, timeout=3600)
    return cache_key


def load_dataframe_efficiently(file_id):
    """Load DataFrame from cache"""
    cache_key = f'dataframe_{file_id}'
    df_pickle = cache.get(cache_key)
    if df_pickle:
        return pickle.loads(df_pickle)
    return None


def dataframe_to_html_paginated(df, page=1, per_page=50):
    """Convert DataFrame to HTML with pagination"""
    total_rows = len(df)
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_rows)
    
    df_page = df.iloc[start_idx:end_idx]
    
    html = df_page.to_html(
        classes='table table-striped table-hover table-sm',
        border=0,
        index=True,
        max_rows=per_page
    )
    
    return {
        'html': html,
        'total_rows': total_rows,
        'page': page,
        'per_page': per_page,
        'total_pages': (total_rows + per_page - 1) // per_page,
        'has_next': end_idx < total_rows,
        'has_prev': page > 1,
        'start_idx': start_idx + 1,
        'end_idx': end_idx
    }


def limit_dataframe_for_display(df, max_size=100, force_full=False):
    """Limit DataFrame size for HTML display"""
    if force_full:
        return df, False
    if len(df) > max_size or len(df.columns) > max_size:
        return df.iloc[:max_size, :max_size], True
    return df, False


# ==============================================================================
# VIEWS
# ==============================================================================

def upload_file(request):
    """Optimized file upload with efficient storage"""
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save(commit=False)
            uploaded_file.filename = request.FILES['file'].name
            uploaded_file.save()
            
            try:
                file_path = uploaded_file.file.path
                
                # Read file with low_memory option for large files
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, low_memory=False)
                else:
                    df = pd.read_excel(file_path)
                
                

                # Check file size and warn user
                if len(df) > 100000:
                    messages.warning(
                        request, 
                        f'Large dataset detected ({len(df):,} rows). '
                        'Some operations may take longer. Consider filtering data.'
                    )
                

                # Store using efficient pickle method in cache
                store_dataframe_efficiently(df, uploaded_file.id)
                
                # Store minimal info in session (not the whole DataFrame!)
                request.session['file_id'] = uploaded_file.id
                request.session['columns'] = list(df.columns)
                request.session['shape'] = list(df.shape)
                request.session['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
                request.session.modified = True
                
                messages.success(
                    request, 
                    f'File "{uploaded_file.filename}" uploaded successfully! '
                    f'({len(df):,} rows × {len(df.columns)} columns)'
                )
                return redirect('dashboard')
            
            except Exception as e:
                messages.error(request, f'Error reading file: {str(e)}')
                uploaded_file.delete()
    else:
        form = FileUploadForm()
    
    return render(request, 'analysis/upload.html', {'form': form})


def dashboard(request):
    """Optimized dashboard with pagination"""
    if 'file_id' not in request.session:
        messages.warning(request, 'Please upload a file first.')
        return redirect('upload_file')
    
    file_id = request.session.get('file_id')
    df = load_dataframe_efficiently(file_id)
    
    if df is None:
        messages.error(request, 'Data expired. Please upload file again.')
        request.session.flush()
        return redirect('upload_file')
    
    # Pagination for data preview
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 25))
    
    paginated_data = dataframe_to_html_paginated(df, page, per_page)
    
    context = {
        'columns': request.session.get('columns', []),
        'data_preview': paginated_data['html'],
        'pagination': paginated_data,
        'shape': request.session.get('shape', [0, 0]),
        'dtypes': request.session.get('dtypes', {}),
    }
    
    return render(request, 'analysis/dashboard.html', context)


def analyze_data(request):
    """Optimized analysis with result caching"""
    if request.method != 'POST':
        return redirect('dashboard')
    
    if 'file_id' not in request.session:
        messages.warning(request, 'Please upload a file first.')
        return redirect('upload_file')
    
    # Get parameters
    selected_columns = request.POST.getlist('columns')
    analysis_type = request.POST.get('analysis_type')
    force_full_matrix = request.POST.get('force_full_matrix') == 'true'
    
    if not selected_columns or not analysis_type:
        messages.error(request, 'Please select columns and analysis type.')
        return redirect('dashboard')
    
    # Generate cache key for this specific analysis
    file_id = request.session.get('file_id')
    cache_suffix = '_full' if force_full_matrix else ''
    cache_key = get_cache_key('analysis', file_id, analysis_type, *selected_columns) + cache_suffix
    
    # Check cache first - MAJOR PERFORMANCE WIN!
    cached_result = cache.get(cache_key)
    if cached_result:
        messages.info(request, '✓ Loaded cached results (instant!)')
        return render(request, 'analysis/results.html', cached_result)
    
    # Load data from cache
    df = load_dataframe_efficiently(file_id)
    if df is None:
        messages.error(request, 'Data expired. Please upload file again.')
        return redirect('upload_file')
    
    # Sample large datasets to improve performance
    original_size = len(df)
    if len(df) > 10000:
        messages.warning(
            request,
            f'Large dataset detected ({original_size:,} rows). '
            f'Using random sample of 10,000 rows for faster analysis.'
        )
        df_subset = df[selected_columns].sample(n=10000, random_state=42)
    else:
        df_subset = df[selected_columns]
    # Record size before drop and drop rows with NaN values
    pre_drop_rows = len(df_subset)
    pre_drop_cols = len(df_subset.columns)
    df_subset = df_subset.dropna(axis=0)
    cleaned_subset_rows = len(df_subset)
    dropped_rows = pre_drop_rows - cleaned_subset_rows
    cleaned_subset_cols = len(df_subset.columns)
    
    # Compute general statistics after drop
    stats = {
        'original_rows': pre_drop_rows,
        'cleaned_rows': cleaned_subset_rows,
        'dropped_rows': dropped_rows,
        'total_columns': cleaned_subset_cols,
        'rows_deleted_pct': round((dropped_rows / pre_drop_rows * 100), 2) if pre_drop_rows > 0 else 0,
    }
    
    # Add per-column statistics
    col_stats = []
    for col in df_subset.columns:
        col_data = df_subset[col]
        col_info = {
            'name': col,
            'dtype': str(col_data.dtype),
            'non_null': col_data.notna().sum(),
            'null': col_data.isna().sum(),
            'unique': col_data.nunique(),
        }
        # Add numeric stats if applicable
        if pd.api.types.is_numeric_dtype(col_data):
            col_info['mean'] = round(col_data.mean(), 4)
            col_info['median'] = round(col_data.median(), 4)
            col_info['mode'] = round(col_data.mode()[0], 4) if len(col_data.mode()) > 0 else 'N/A'
            col_info['std'] = round(col_data.std(), 4)
            col_info['var'] = round(col_data.var(), 4)
            col_info['min'] = round(col_data.min(), 4)
            col_info['q1'] = round(col_data.quantile(0.25), 4)
            col_info['q3'] = round(col_data.quantile(0.75), 4)
            col_info['max'] = round(col_data.max(), 4)
            col_info['iqr'] = round(col_data.quantile(0.75) - col_data.quantile(0.25), 4)
            col_info['skewness'] = round(col_data.skew(), 4)
            col_info['kurtosis'] = round(col_data.kurtosis(), 4)
        else:
            # For categorical columns, show mode
            mode_val = col_data.mode()
            col_info['mode'] = mode_val[0] if len(mode_val) > 0 else 'N/A'
        col_stats.append(col_info)
    stats['columns'] = col_stats
    
    # Prepare a limited HTML preview for display
    _clean_preview, _was_limited = limit_dataframe_for_display(df_subset, max_size=100, force_full=False)
    cleaned_subset_html = _clean_preview.to_html(
        classes='table table-striped table-hover table-sm',
        border=0,
        index=True
    )
    
    # Ordinal configuration only for distance/dissimilarity analyses
    if analysis_type in ('distance', 'dissimilarity'):
        # Build uniques map for ordinal configuration
        uniques_map = {}
        for col in selected_columns:
            vals = df_subset[col].dropna().astype(str)
            freq = vals.value_counts()
            max_uniques = 200
            if len(freq) > max_uniques:
                messages.warning(request, f'Column "{col}" has {len(freq)} categories. Showing top {max_uniques} by frequency.')
                freq = freq.head(max_uniques)
            uniques_map[col] = list(freq.index)

        # If ordinal step not confirmed, render ordinal configuration page
        if request.POST.get('ordinal_confirmed') != 'true':
            uniques_items = [(col, uniques_map.get(col, [])) for col in selected_columns]
            return render(request, 'analysis/ordinal.html', {
                'selected_columns': selected_columns,
                'analysis_type': analysis_type,
                'force_full_matrix': force_full_matrix,
                'uniques_items': uniques_items,
                'stats': stats,
            })

        # Apply user-provided ordinal ordering
        for col in selected_columns:
            if request.POST.get(f'is_ordinal_{col}') == 'yes':
                raw = request.POST.get(f'order_{col}', '')
                order_list = [line.strip() for line in raw.splitlines() if line.strip()]
                existing = set(df_subset[col].astype(str).unique())
                order_list = [v for v in order_list if v in existing]
                if order_list:
                    df_subset[col] = pd.Categorical(df_subset[col].astype(str), categories=order_list, ordered=True)

    # Initialize backend with your existing DataAnalysisBackend
    backend = DataAnalysisBackend(df_subset.to_dict('list'))
    
    results = {}
    # Include cleaned subset preview, counts, and statistics in results for optional display
    results['cleaned_subset_html'] = cleaned_subset_html
    results['cleaned_subset_rows'] = cleaned_subset_rows
    results['dropped_rows'] = dropped_rows
    results['stats'] = stats
    result_df = None
    
    try:
        if analysis_type == 'contingency' and len(selected_columns) >= 2:
            table = backend.compute_contingency_table(
                selected_columns[0], 
                selected_columns[1]
            )
            result_df = table
            results['title'] = f'Contingency Table: {selected_columns[0]} vs {selected_columns[1]}'
            
        elif analysis_type == 'burt':
            X = backend.get_disjunctive_coding()
            result_df = backend.compute_burt_matrix(X)
            results['title'] = 'Burt Matrix'
            
        elif analysis_type == 'distance':
            X = backend.get_disjunctive_coding()
            individuals = list(range(len(df_subset)))
            dist_df, frac_df = backend.compute_distance_matrix(
                X.to_numpy(), 
                individuals
            )
            # Limit display for large matrices
            limited_frac, was_limited = limit_dataframe_for_display(frac_df, max_size=100, force_full=force_full_matrix)
            if was_limited and not force_full_matrix:
                results['table'] = (
                    '<div class="alert alert-warning">'
                    '<strong>Matrix too large</strong> - showing first 100×100. '
                    '<form method="post" action="" style="display:inline;">'
                    f'<input type="hidden" name="csrfmiddlewaretoken" value="{request.POST.get("csrfmiddlewaretoken")}">'
                    f'<input type="hidden" name="analysis_type" value="{analysis_type}">'
                )
                for col in selected_columns:
                    results['table'] += f'<input type="hidden" name="columns" value="{col}">'
                results['table'] += (
                    '<input type="hidden" name="force_full_matrix" value="true">'
                    '<button type="submit" class="btn btn-sm btn-warning ms-2">'
                    '<i class="bi bi-grid-3x3-gap me-1"></i>Load Full Matrix</button>'
                    '</form>'
                    '</div>' +
                    limited_frac.to_html(classes='table table-bordered table-sm', border=0)
                )
                results['show_full_warning'] = True
            elif force_full_matrix:
                messages.warning(request, f'Loading full matrix ({len(frac_df)}×{len(frac_df.columns)}). This may take a moment...')
                results['table'] = frac_df.to_html(classes='table table-bordered table-sm', border=0)
            else:
                results['table'] = frac_df.to_html(classes='table table-bordered table-sm', border=0)
            
            result_df = dist_df
            results['title'] = 'Distance Matrix'
            
        elif analysis_type == 'dissimilarity':
            X = backend.get_disjunctive_coding()
            individuals = list(range(len(df_subset)))
            sim_df, frac_df = backend.compute_dissimilarity_matrix(
                X.to_numpy(), 
                individuals
            )
            limited_frac, was_limited = limit_dataframe_for_display(frac_df, max_size=100, force_full=force_full_matrix)
            if was_limited and not force_full_matrix:
                results['table'] = (
                    '<div class="alert alert-warning">'
                    '<strong>Matrix too large</strong> - showing first 100×100. '
                    '<form method="post" action="" style="display:inline;">'
                    f'<input type="hidden" name="csrfmiddlewaretoken" value="{request.POST.get("csrfmiddlewaretoken")}">'
                    f'<input type="hidden" name="analysis_type" value="{analysis_type}">'
                )
                for col in selected_columns:
                    results['table'] += f'<input type="hidden" name="columns" value="{col}">'
                results['table'] += (
                    '<input type="hidden" name="force_full_matrix" value="true">'
                    '<button type="submit" class="btn btn-sm btn-warning ms-2">'
                    '<i class="bi bi-grid-3x3-gap me-1"></i>Load Full Matrix</button>'
                    '</form>'
                    '</div>' +
                    limited_frac.to_html(classes='table table-bordered table-sm', border=0)
                )
                results['show_full_warning'] = True
            elif force_full_matrix:
                messages.warning(request, f'Loading full matrix ({len(frac_df)}×{len(frac_df.columns)}). This may take a moment...')
                results['table'] = frac_df.to_html(classes='table table-bordered table-sm', border=0)
            else:
                results['table'] = frac_df.to_html(classes='table table-bordered table-sm', border=0)
            
            result_df = sim_df
            results['title'] = 'Dissimilarity Matrix'
            
        elif analysis_type == 'row_profiles' and len(selected_columns) >= 2:
            table = backend.compute_contingency_table(
                selected_columns[0], 
                selected_columns[1]
            )
            row_prof, cloud_display = backend.compute_row_profiles(table)
            results['table'] = cloud_display.to_html(classes='table table-bordered', border=0)
            result_df = row_prof
            results['title'] = 'Row Profiles'
            
        elif analysis_type == 'col_profiles' and len(selected_columns) >= 2:
            table = backend.compute_contingency_table(
                selected_columns[0], 
                selected_columns[1]
            )
            col_prof, col_cloud_display = backend.compute_column_profiles(table)
            results['table'] = col_cloud_display.to_html(classes='table table-bordered', border=0)
            result_df = col_prof
            results['title'] = 'Column Profiles'
            
        elif analysis_type == 'chi2_rows' and len(selected_columns) >= 2:
            table = backend.compute_contingency_table(
                selected_columns[0], 
                selected_columns[1]
            )
            result_df = backend.compute_chi2_distance_rows(table)
            results['title'] = 'Chi-Square Distance (Rows)'
            
        elif analysis_type == 'chi2_cols' and len(selected_columns) >= 2:
            table = backend.compute_contingency_table(
                selected_columns[0], 
                selected_columns[1]
            )
            result_df = backend.compute_chi2_distance_cols(table)
            results['title'] = 'Chi-Square Distance (Columns)'
        
        else:
            messages.error(
                request, 
                'Invalid analysis type or insufficient columns. Some analyses require 2+ columns.'
            )
            return redirect('dashboard')
        
        # Generate HTML table if not already set
        if 'table' not in results and result_df is not None:
            limited_df, was_limited = limit_dataframe_for_display(result_df, max_size=100, force_full=force_full_matrix)
            
            if was_limited and not force_full_matrix:
                results['table'] = (
                    '<div class="alert alert-warning">'
                    '<strong>Result too large</strong> - showing first 100×100. '
                    '<form method="post" action="" style="display:inline;">'
                    f'<input type="hidden" name="csrfmiddlewaretoken" value="{request.POST.get("csrfmiddlewaretoken")}">'
                    f'<input type="hidden" name="analysis_type" value="{analysis_type}">'
                )
                for col in selected_columns:
                    results['table'] += f'<input type="hidden" name="columns" value="{col}">'
                results['table'] += (
                    '<input type="hidden" name="force_full_matrix" value="true">'
                    '<button type="submit" class="btn btn-sm btn-warning ms-2">'
                    '<i class="bi bi-grid-3x3-gap me-1"></i>Load Full Matrix</button>'
                    '</form>'
                    '</div>' +
                    limited_df.to_html(
                        classes='table table-bordered table-sm',
                        border=0,
                        float_format=lambda x: f'{x:.4f}' if abs(x) < 1000 else f'{x:.2e}'
                    )
                )
                results['show_full_warning'] = True
            elif force_full_matrix:
                messages.warning(request, f'Loading full result ({len(result_df)}×{len(result_df.columns)}). This may take a moment...')
                results['table'] = result_df.to_html(
                    classes='table table-bordered table-sm',
                    border=0,
                    float_format=lambda x: f'{x:.4f}' if abs(x) < 1000 else f'{x:.2e}'
                )
            else:
                results['table'] = result_df.to_html(
                    classes='table table-bordered table-sm',
                    border=0,
                    float_format=lambda x: f'{x:.4f}' if abs(x) < 1000 else f'{x:.2e}'
                )
        
        results['selected_columns'] = selected_columns
        results['analysis_type'] = analysis_type
        
        # Store result for graphing (with expiry) using pickle
        if result_df is not None:
            result_cache_key = get_cache_key('result', file_id, analysis_type, *selected_columns)
            result_pickle = pickle.dumps(result_df)
            cache.set(result_cache_key, result_pickle, timeout=1800)  # 30 min
            request.session['current_result_key'] = result_cache_key
            request.session.modified = True
        
        # Cache the full results for 30 minutes - MAJOR PERFORMANCE WIN!
        cache.set(cache_key, results, timeout=1800)
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        messages.error(request, f'Analysis error: {str(e)}')
        print(f"Error details:\n{error_trace}")  # For debugging
        return redirect('dashboard')
    
    return render(request, 'analysis/results.html', results)


@csrf_exempt
def generate_graph(request):
    """Optimized graph generation with caching"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
    result_key = request.session.get('current_result_key')
    if not result_key:
        return JsonResponse({
            'error': 'No data available. Please run an analysis first.'
        }, status=400)
    
    # Get cached result
    result_pickle = cache.get(result_key)
    if not result_pickle:
        return JsonResponse({
            'error': 'Result data expired. Please run the analysis again.'
        }, status=400)
    
    try:
        df = pickle.loads(result_pickle)
        
        # Get parameters
        graph_type = request.POST.get('graph_type', 'heatmap')
        title = request.POST.get('title', 'Data Visualization')
        xlabel = request.POST.get('xlabel', '')
        ylabel = request.POST.get('ylabel', '')
        colormap = request.POST.get('colormap', 'viridis')
        figsize_w = int(request.POST.get('figsize_width', 10))
        figsize_h = int(request.POST.get('figsize_height', 8))
        
        # Check cache for this specific graph configuration
        graph_cache_key = get_cache_key(
            'graph', result_key, graph_type, title, colormap, figsize_w, figsize_h
        )
        cached_image = cache.get(graph_cache_key)
        
        if cached_image:
            return JsonResponse({
                'success': True,
                'image': cached_image,
                'cached': True
            })
        
        # Limit data size for visualization (performance improvement)
        if len(df) > 100:
            df = df.iloc[:100, :100] if hasattr(df, 'iloc') else df
            messages.info(request, 'Large result - visualizing first 100×100 subset')
        
        # Create figure with lower DPI for faster rendering
        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h), dpi=100)
        
        if graph_type == 'heatmap':
            df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Optimize annotation for large matrices
            annot = len(df_numeric) * len(df_numeric.columns) < 500
            
            sns.heatmap(
                df_numeric, 
                annot=annot,
                fmt='.2f' if annot else '',
                cmap=colormap,
                cbar=True,
                ax=ax,
                square=False,
                cbar_kws={'shrink': 0.8}
            )
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
        elif graph_type == 'bar':
            df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Limit bars for readability
            if len(df_numeric) > 50:
                df_numeric = df_numeric.iloc[:50]
                xlabel = f'{xlabel} (first 50 shown)'
            
            if len(df_numeric.shape) == 2 and df_numeric.shape[1] < 10:
                df_numeric.plot(kind='bar', ax=ax)
                ax.legend(loc='best', fontsize=8)
            else:
                df_numeric.iloc[:, 0].plot(kind='bar', ax=ax)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel if xlabel else 'Categories')
            ax.set_ylabel(ylabel if ylabel else 'Values')
            ax.tick_params(axis='x', rotation=45)
            
        elif graph_type == 'line':
            df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            df_numeric.plot(kind='line', ax=ax, marker='o', markersize=3)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel if xlabel else 'Index')
            ax.set_ylabel(ylabel if ylabel else 'Values')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
        elif graph_type == 'scatter' and df.shape[1] >= 2:
            df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            ax.scatter(df_numeric.iloc[:, 0], df_numeric.iloc[:, 1], alpha=0.6, s=50)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel if xlabel else df.columns[0])
            ax.set_ylabel(ylabel if ylabel else df.columns[1])
            ax.grid(True, alpha=0.3)
            
        elif graph_type == 'box':
            df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            df_numeric.plot(kind='box', ax=ax)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel(ylabel if ylabel else 'Values')
            
        elif graph_type == 'histogram':
            df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            df_numeric.plot(kind='hist', alpha=0.7, bins=20, ax=ax)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel if xlabel else 'Values')
            ax.set_ylabel(ylabel if ylabel else 'Frequency')
            ax.legend(loc='best', fontsize=8)
            
        elif graph_type == 'pie':
            df_numeric = df.iloc[:, 0].apply(pd.to_numeric, errors='coerce').fillna(0)
            df_numeric = df_numeric[df_numeric > 0]
            if len(df_numeric) > 0:
                # Limit to top 10 slices
                if len(df_numeric) > 10:
                    df_numeric = df_numeric.nlargest(10)
                ax.pie(df_numeric, labels=df_numeric.index, autopct='%1.1f%%', startangle=90)
                ax.set_title(title, fontsize=14, fontweight='bold')
            else:
                return JsonResponse({'error': 'No positive values for pie chart'}, status=400)
        
        else:
            return JsonResponse({
                'error': f'Graph type "{graph_type}" not supported or requires different data'
            }, status=400)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        image_data = f'data:image/png;base64,{image_base64}'
        
        # Cache the generated image for 30 minutes
        cache.set(graph_cache_key, image_data, timeout=1800)
        
        return JsonResponse({
            'success': True,
            'image': image_data
        })
        
    except Exception as e:
        import traceback
        return JsonResponse({
            'error': f'Graph error: {str(e)}',
            'details': traceback.format_exc()
        }, status=500)


def clear_session(request):
    """Clear session and cache"""
    file_id = request.session.get('file_id')
    if file_id:
        # Clear cached data
        cache.delete(f'dataframe_{file_id}')
    
    request.session.flush()
    messages.info(request, 'Session cleared. Please upload a new file.')
    return redirect('upload_file')