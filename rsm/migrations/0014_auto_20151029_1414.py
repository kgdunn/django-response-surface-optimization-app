# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0013_auto_20151027_1528'),
    ]

    operations = [
        migrations.AddField(
            model_name='input',
            name='plot_lower_bound',
            field=models.FloatField(default=0.0, help_text=b'Plots must be generated to show the true solution. What is the lower used in these plots for this variable?'),
        ),
        migrations.AddField(
            model_name='input',
            name='plot_upper_bound',
            field=models.FloatField(default=0.0, help_text=b'Plots must be generated to show the true solution. What is the upper used in these plots for this variable?'),
        ),
        migrations.AlterField(
            model_name='input',
            name='default_value',
            field=models.FloatField(help_text=b'The default used, e.g. in a multidimensional (>3) plot. For categorical variables this MUST correspond to one of the levels in the JSON dictionary.'),
        ),
        migrations.AlterField(
            model_name='input',
            name='level_numeric_mapping',
            field=models.TextField(help_text=b'For example: {"water": -1, "vinegar": +1} would map the "water" level to -1 and "vinegar" level to +1 in the simulation. Leave blank for continuous variables.', verbose_name=b'Specify UNIQUE names for each numeric level of a categorical variable; JSON format.', blank=True),
        ),
        migrations.AlterField(
            model_name='input',
            name='n_decimals',
            field=models.PositiveSmallIntegerField(default=0, help_text=b'The number of decimals to show in the numeric representation. Not applicable for categorical variables (leave as 0.0)', verbose_name=b'Number of decimals'),
        ),
    ]
