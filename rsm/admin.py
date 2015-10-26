from django.contrib import admin


from .models import Person, Token, Tag, Experiment, System, Input

class SystemAdmin(admin.ModelAdmin):
    prepopulated_fields = {"slug": ("full_name",)}

class InputAdmin(admin.ModelAdmin):
    prepopulated_fields = {"slug": ("display_name",)}

admin.site.register(Person)
admin.site.register(Token)
admin.site.register(Tag)
admin.site.register(Experiment)
admin.site.register(System, SystemAdmin)
admin.site.register(Input, InputAdmin)