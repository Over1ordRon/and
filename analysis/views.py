# =============================================================================
# analysis/views.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json

from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from .models import DataFile
from .forms import FileUploadForm, ColumnSelectionForm, AnalysisTypeForm, GraphConfigForm
from .backend import DataAnalysisBackend


def upload_file(request):
    """View for uploading CSV/Excel files"""
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save(commit=False)
            uploaded_file.filename = request.FILES['file'].name
            uploaded_file.save()
            
            # Read file and store in session
            try:
                file_path = uploaded_file.file.path
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                # Store data in session
                request.session['data'] = df.to_json(orient='split', date_format='iso')
                request.session['columns'] = list(df.columns)
                request.session['file_id'] = uploaded_file.id
                request.session.modified = True
                
                messages.success(request, f'File "{uploaded_file.filename}" uploaded successfully!')
                return redirect('dashboard')
            
            except Exception as e:
                messages.error(request, f'Error reading file: {str(e)}')
                uploaded_file.delete()
    else:
        form = FileUploadForm()
    
    return render(request, 'analysis/upload.html', {'form': form})


def dashboard(request):
    """Main dashboard for data analysis"""
    if 'data' not in request.session:
        messages.warning(request, 'Please upload a file first.')
        return redirect('upload_file')
    
    # Load data from session
    data_json = request.session.get('data')
    df = pd.read_json(data_json, orient='split')
    columns = request.session.get('columns', [])
    
    # Get data types as strings
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    context = {
        'columns': columns,
        'data_preview': df.head(10).to_html(classes='table table-striped table-hover', border=0),
        'shape': df.shape,
        'dtypes': dtypes_dict,
    }
    
    return render(request, 'analysis/dashboard.html', context)


def analyze_data(request):
    """Perform analysis on selected columns"""
    if request.method != 'POST':
        return redirect('dashboard')
    
    if 'data' not in request.session:
        messages.warning(request, 'Please upload a file first.')
        return redirect('upload_file')
    
    # Get selected columns
    selected_columns = request.POST.getlist('columns')
    analysis_type = request.POST.get('analysis_type')
    
    if not selected_columns:
        messages.error(request, 'Please select at least one column.')
        return redirect('dashboard')
    
    if not analysis_type:
        messages.error(request, 'Please select an analysis type.')
        return redirect('dashboard')
    
    # Load data
    data_json = request.session.get('data')
    df = pd.read_json(data_json, orient='split')
    df_subset = df[selected_columns]
    
    # Initialize backend
    backend = DataAnalysisBackend(df_subset.to_dict('list'))
    
    # Perform analysis based on type
    results = {}
    
    try:
        if analysis_type == 'contingency' and len(selected_columns) >= 2:
            table = backend.compute_contingency_table(selected_columns[0], selected_columns[1])
            results['table'] = table.to_html(classes='table table-bordered', border=0)
            results['title'] = f'Contingency Table: {selected_columns[0]} vs {selected_columns[1]}'
            request.session['current_table'] = table.to_json(orient='split')
            
        elif analysis_type == 'burt':
            X = backend.get_disjunctive_coding()
            burt = backend.compute_burt_matrix(X)
            results['table'] = burt.to_html(classes='table table-bordered', border=0)
            results['title'] = 'Burt Matrix'
            request.session['current_table'] = burt.to_json(orient='split')
            
        elif analysis_type == 'distance':
            X = backend.get_disjunctive_coding()
            individuals = list(range(len(df_subset)))
            dist_df, frac_df = backend.compute_distance_matrix(X.to_numpy(), individuals)
            results['table'] = frac_df.to_html(classes='table table-bordered', border=0)
            results['numeric_table'] = dist_df.to_html(classes='table table-bordered', border=0)
            results['title'] = 'Distance Matrix'
            request.session['current_table'] = dist_df.to_json(orient='split')
            
        elif analysis_type == 'dissimilarity':
            X = backend.get_disjunctive_coding()
            individuals = list(range(len(df_subset)))
            sim_df, frac_df = backend.compute_dissimilarity_matrix(X.to_numpy(), individuals)
            results['table'] = frac_df.to_html(classes='table table-bordered', border=0)
            results['numeric_table'] = sim_df.to_html(classes='table table-bordered', border=0)
            results['title'] = 'Dissimilarity Matrix'
            request.session['current_table'] = sim_df.to_json(orient='split')
            
        elif analysis_type == 'row_profiles' and len(selected_columns) >= 2:
            table = backend.compute_contingency_table(selected_columns[0], selected_columns[1])
            row_prof, cloud_display = backend.compute_row_profiles(table)
            results['table'] = cloud_display.to_html(classes='table table-bordered', border=0)
            results['numeric_table'] = row_prof.to_html(classes='table table-bordered', border=0)
            results['title'] = 'Row Profiles'
            request.session['current_table'] = row_prof.to_json(orient='split')
            
        elif analysis_type == 'col_profiles' and len(selected_columns) >= 2:
            table = backend.compute_contingency_table(selected_columns[0], selected_columns[1])
            col_prof, col_cloud_display = backend.compute_column_profiles(table)
            results['table'] = col_cloud_display.to_html(classes='table table-bordered', border=0)
            results['numeric_table'] = col_prof.to_html(classes='table table-bordered', border=0)
            results['title'] = 'Column Profiles'
            request.session['current_table'] = col_prof.to_json(orient='split')
            
        elif analysis_type == 'chi2_rows' and len(selected_columns) >= 2:
            table = backend.compute_contingency_table(selected_columns[0], selected_columns[1])
            chi2_dist = backend.compute_chi2_distance_rows(table)
            results['table'] = chi2_dist.to_html(classes='table table-bordered', border=0)
            results['title'] = 'Chi-Square Distance (Rows)'
            request.session['current_table'] = chi2_dist.to_json(orient='split')
            
        elif analysis_type == 'chi2_cols' and len(selected_columns) >= 2:
            table = backend.compute_contingency_table(selected_columns[0], selected_columns[1])
            chi2_dist = backend.compute_chi2_distance_cols(table)
            results['table'] = chi2_dist.to_html(classes='table table-bordered', border=0)
            results['title'] = 'Chi-Square Distance (Columns)'
            request.session['current_table'] = chi2_dist.to_json(orient='split')
        
        else:
            messages.error(request, 'Invalid analysis type or insufficient columns selected. Some analyses require 2+ columns.')
            return redirect('dashboard')
        
        results['selected_columns'] = selected_columns
        results['analysis_type'] = analysis_type
        request.session.modified = True
        
    except Exception as e:
        messages.error(request, f'Analysis error: {str(e)}')
        return redirect('dashboard')
    
    return render(request, 'analysis/results.html', results)


@csrf_exempt
def generate_graph(request):
    """Generate matplotlib graph based on current table"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
    if 'current_table' not in request.session:
        return JsonResponse({'error': 'No data available for graphing. Please run an analysis first.'}, status=400)
    
    try:
        # Load current table
        table_json = request.session.get('current_table')
        df = pd.read_json(table_json, orient='split')
        
        # Get graph configuration
        graph_type = request.POST.get('graph_type', 'heatmap')
        title = request.POST.get('title', 'Data Visualization')
        xlabel = request.POST.get('xlabel', '')
        ylabel = request.POST.get('ylabel', '')
        colormap = request.POST.get('colormap', 'viridis')
        figsize_w = int(request.POST.get('figsize_width', 10))
        figsize_h = int(request.POST.get('figsize_height', 8))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
        
        if graph_type == 'heatmap':
            # Convert to numeric, handling any non-numeric values
            df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            sns.heatmap(df_numeric, annot=True, fmt='.2f', cmap=colormap, cbar=True, ax=ax)
            ax.set_title(title, fontsize=14, fontweight='bold')
            
        elif graph_type == 'bar':
            if len(df.shape) == 2 and df.shape[0] < 20 and df.shape[1] < 10:
                df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                df_numeric.plot(kind='bar', ax=ax)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel(xlabel if xlabel else 'Categories')
                ax.set_ylabel(ylabel if ylabel else 'Values')
                ax.tick_params(axis='x', rotation=45)
                ax.legend(loc='best')
            else:
                # Plot first column only
                df_numeric = df.iloc[:, 0].apply(pd.to_numeric, errors='coerce').fillna(0)
                df_numeric.plot(kind='bar', ax=ax)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel(xlabel if xlabel else 'Index')
                ax.set_ylabel(ylabel if ylabel else df.columns[0])
                ax.tick_params(axis='x', rotation=45)
            
        elif graph_type == 'line':
            df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            df_numeric.plot(kind='line', ax=ax, marker='o')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel if xlabel else 'Index')
            ax.set_ylabel(ylabel if ylabel else 'Values')
            ax.legend(loc='best')
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
            ax.legend(loc='best')
            
        elif graph_type == 'pie':
            # Use first column for pie chart
            df_numeric = df.iloc[:, 0].apply(pd.to_numeric, errors='coerce').fillna(0)
            # Filter out zero or negative values
            df_numeric = df_numeric[df_numeric > 0]
            if len(df_numeric) > 0:
                ax.pie(df_numeric, labels=df_numeric.index, autopct='%1.1f%%', startangle=90)
                ax.set_title(title, fontsize=14, fontweight='bold')
            else:
                return JsonResponse({'error': 'No positive values available for pie chart'}, status=400)
        
        else:
            return JsonResponse({'error': f'Graph type "{graph_type}" is not supported or requires different data structure'}, status=400)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return JsonResponse({
            'success': True,
            'image': f'data:image/png;base64,{image_base64}'
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return JsonResponse({
            'error': f'Graph generation error: {str(e)}',
            'details': error_details
        }, status=500)


def clear_session(request):
    """Clear session data"""
    request.session.flush()
    messages.info(request, 'Session cleared. Please upload a new file.')
    return redirect('upload_file')