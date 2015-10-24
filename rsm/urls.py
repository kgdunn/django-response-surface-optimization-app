from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.show_all_systems, name='show_all_systems'),
    
    # ex: /system/short_name/
    url(r'^system/(?P<short_name_slug>[-\w]+)/$', views.show_one_system, 
        name='show_one_system'),    
]