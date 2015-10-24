from django.contrib import admin


from .models import Person, Token, Tag, Result, System

class SystemAdmin(admin.ModelAdmin):
    prepopulated_fields = {"slug": ("full_name",)}

admin.site.register(Person)
admin.site.register(Token)
admin.site.register(Tag)
admin.site.register(Result)
admin.site.register(System, SystemAdmin)