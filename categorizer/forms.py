from django import forms
from categorizer.models import *

class ProfileForm(forms.Form):
   name = forms.CharField(max_length = 100)
   file = forms.FileField()


class JobFileSubmitForm(forms.ModelForm):

    class Meta:
        model = JobFileSubmit
        fields = 'file',

    def save(self, commit=True):
        jobfilesubmit = super(JobFileSubmitForm, self).save(commit=False)

        if commit:
            jobfilesubmit.save()
        return jobfilesubmit