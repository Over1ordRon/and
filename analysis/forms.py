
# =============================================================================
# analysis/forms.py
# =============================================================================

from django import forms
from .models import DataFile

class FileUploadForm(forms.ModelForm):
    """Form for uploading CSV/Excel files"""
    class Meta:
        model = DataFile
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv,.xlsx,.xls'
            })
        }
    
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            ext = file.name.split('.')[-1].lower()
            if ext not in ['csv', 'xlsx', 'xls']:
                raise forms.ValidationError('Only CSV and Excel files are allowed.')
            if file.size > 10 * 1024 * 1024:  # 10MB limit
                raise forms.ValidationError('File size must be under 10MB.')
        return file


class ColumnSelectionForm(forms.Form):
    """Dynamic form for selecting columns to analyze"""
    def __init__(self, columns, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['columns'] = forms.MultipleChoiceField(
            choices=[(col, col) for col in columns],
            widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}),
            required=True,
            label='Select columns to analyze'
        )


class AnalysisTypeForm(forms.Form):
    """Form for selecting analysis type"""
    ANALYSIS_CHOICES = [
        ('contingency', 'Contingency Table'),
        ('burt', 'Burt Matrix'),
        ('distance', 'Distance Matrix'),
        ('dissimilarity', 'Dissimilarity Matrix'),
        ('row_profiles', 'Row Profiles'),
        ('col_profiles', 'Column Profiles'),
        ('chi2_rows', 'Chi-Square Distance (Rows)'),
        ('chi2_cols', 'Chi-Square Distance (Columns)'),
    ]
    
    analysis_type = forms.ChoiceField(
        choices=ANALYSIS_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        required=True
    )


class GraphConfigForm(forms.Form):
    """Form for configuring matplotlib graphs"""
    GRAPH_TYPES = [
        ('bar', 'Bar Chart'),
        ('line', 'Line Plot'),
        ('scatter', 'Scatter Plot'),
        ('heatmap', 'Heatmap'),
        ('box', 'Box Plot'),
        ('histogram', 'Histogram'),
        ('pie', 'Pie Chart'),
    ]
    
    graph_type = forms.ChoiceField(
        choices=GRAPH_TYPES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        required=True,
        label='Graph Type'
    )
    
    title = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Graph Title'}),
        label='Title'
    )
    
    xlabel = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'X-axis Label'}),
        label='X-axis Label'
    )
    
    ylabel = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Y-axis Label'}),
        label='Y-axis Label'
    )
    
    colormap = forms.ChoiceField(
        choices=[
            ('viridis', 'Viridis'),
            ('plasma', 'Plasma'),
            ('inferno', 'Inferno'),
            ('magma', 'Magma'),
            ('coolwarm', 'Coolwarm'),
            ('RdYlBu', 'Red-Yellow-Blue'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'}),
        required=False,
        label='Color Map (for heatmap)'
    )
    
    figsize_width = forms.IntegerField(
        min_value=4,
        max_value=20,
        initial=10,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Figure Width'
    )
    
    figsize_height = forms.IntegerField(
        min_value=4,
        max_value=20,
        initial=8,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Figure Height'
    )