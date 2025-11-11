# =============================================================================
# analysis/models.py
# =============================================================================

from django.db import models
from django.core.validators import FileExtensionValidator

class DataFile(models.Model):
    """Model to store uploaded data files"""
    file = models.FileField(
        upload_to='uploads/%Y/%m/%d/',
        validators=[FileExtensionValidator(allowed_extensions=['csv', 'xlsx', 'xls'])]
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    filename = models.CharField(max_length=255)
    
    def __str__(self):
        return f"{self.filename} - {self.uploaded_at}"
    
    class Meta:
        ordering = ['-uploaded_at']

