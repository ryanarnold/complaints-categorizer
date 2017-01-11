from django.db import models
from django.core.files.storage import FileSystemStorage

# Create your models here.
fs = FileSystemStorage(location='/media/')

def upload_job_file_path(instance, filename):
    return 'data/%s' % (filename)


class JobFileSubmit(models.Model):
    file = models.FileField(upload_to=upload_job_file_path, null=False)
    uploadDate = models.DateTimeField(auto_now=True)
    
class Profile(models.Model):
   name = models.CharField(max_length = 50)
   file = models.FileField(upload_to = 'data')

   class Meta:
      db_table = "complaintfile"