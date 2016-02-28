from django.contrib import admin


from .models import Person, Token, Tag, Experiment, System, Input, PersonSystem

class SystemAdmin(admin.ModelAdmin):
    prepopulated_fields = {"slug": ("full_name",)}
    list_display = ('full_name', 'slug', 'is_active', 'n_inputs', 'n_outputs',
                    'cost_per_experiment', 'max_experiments_allowed')
    list_display_links = list_display

class PersonSystemAdmin(admin.ModelAdmin):

    list_display = ('person', 'system', 'rotation', 'offset_y',
                    'completed_date', 'show_solution_as_of', 'frozen',
                    'started_on')
    list_display_links = list_display

class InputAdmin(admin.ModelAdmin):
    prepopulated_fields = {"slug": ("display_name",)}
    list_display = ('display_name', 'slug', 'system', 'ntype', 'lower_bound',
                    'plot_lower_bound', 'upper_bound', 'plot_upper_bound',
                    'units_prefix', 'default_value', 'units_suffix',
                    'n_decimals')
    list_display_links = list_display

class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('person', 'system', 'time_to_solve', 'earliest_to_show',
                    'main_result', 'was_successful')
    list_display_links = ('person', 'system', 'time_to_solve',
                          'earliest_to_show')

class PersonAdmin(admin.ModelAdmin):
    list_display = ('display_name', 'email', 'level', 'is_validated',)
    list_display_links = list_display

class TokenAdmin(admin.ModelAdmin):
    list_display = ('person', 'system', 'was_used', 'time_used',
                    'next_URI', 'experiment',)
    list_display_links = list_display

class PlotHashAdmin(admin.ModelAdmin):
    list_display = ('person', 'system', 'time_last_used', 'plot_HTML',
                    )
    list_display_links = list_display

admin.site.register(Tag)
admin.site.register(Person, PersonAdmin)
admin.site.register(Token, TokenAdmin)
admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(System, SystemAdmin)
admin.site.register(Input, InputAdmin)
admin.site.register(PersonSystem, PersonSystemAdmin)