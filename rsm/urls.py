from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.show_all_systems, name='show_all_systems'),

    # ex: /system/short_name/
    url(r'^system/(?P<short_name_slug>[-\w]+)/$', views.show_one_system,
        name='show_one_system'),

    # Example: /validate/asd1241a
    url(r'^validate/(?P<hashvalue>[-\w]+)/$', views.validate_user,
        name='validate_user'),

    # Example: /sign-in/asd1241a
    url(r'^sign-in/(?P<hashvalue>[-\w]+)/$', views.sign_in_user,
            name='sign_in_user'),

    # Example: /web-sign-in/
       url(r'^popup-sign-in$', views.popup_sign_in, name='popup_sign_in'),


]