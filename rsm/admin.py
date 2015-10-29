from django.contrib import admin


from .models import Person, Token, Tag, Experiment, System, Input

class SystemAdmin(admin.ModelAdmin):
    prepopulated_fields = {"slug": ("full_name",)}

class InputAdmin(admin.ModelAdmin):
    prepopulated_fields = {"slug": ("display_name",)}

class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('person', 'system', 'time_to_solve', 'earliest_to_show',
                    'is_valid', 'main_result')
    list_display_links = ('person', 'system', 'time_to_solve',
                          'earliest_to_show')
    list_filter = (
        ('is_valid', admin.BooleanFieldListFilter),
    )

admin.site.register(Person)
admin.site.register(Token)
admin.site.register(Tag)
admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(System, SystemAdmin)
admin.site.register(Input, InputAdmin)