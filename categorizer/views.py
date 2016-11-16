from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse

def index(request):
    return HttpResponseRedirect(reverse('home'))

def home(request):
    return render(request, 'home.html')

def categorizer(request):
    return render(request, 'categorizer.html')

def categorized(request):
    return render(request, 'categorized.html')

def complaints(request):
    return render(request, 'complaints.html')
    
def adminpage(request):
    return render(request, 'adminpage.html')
    
def data(request):
    return render(request, 'data.html')
    
def performance(request):
    return render(request, 'performance.html')
