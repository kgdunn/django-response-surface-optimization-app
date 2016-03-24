from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve

from . import views

urlpatterns = [

    # ex: /   (base URL)
    url(r'^$', views.show_all_systems, name='show_all_systems'),

    # ex: /system/short_name/
    url(r'^system/(?P<short_name_slug>[-\w]+)/$', views.show_one_system,
        name='show_one_system'),

    # ex: /system/short_name/reset
    url(r'^system/(?P<short_name_slug>[-\w]+)/reset$', views.reset_one_system,
        name='reset_one_system'),

    # ex: /system/short_name/show-solution
    url(r'^system/(?P<short_name_slug>[-\w]+)/show-solution$',
        views.show_solution_one_system, name='show_solution_one_system'),

    # ex: /system/short_name/other/"user-slug"
        url(r'^system/(?P<short_name_slug>[-\w]+)/other/(?P<other_slug>[-\w]+)$',
            views.show_one_system_other, name='show_one_system_other'),

    # Example: /validate/HGSAT
    url(r'^validate/(?P<hashvalue>[-\w]+)/$', views.validate_user,
        name='validate_user'),

    # Example: /sign-in/QURAA
    url(r'^sign-in/(?P<hashvalue>[-\w]+)/$', views.sign_in_user,
            name='sign_in_user'),

    # Example: /web-sign-in/
       url(r'^popup-sign-in$', views.popup_sign_in, name='popup_sign_in'),


]
