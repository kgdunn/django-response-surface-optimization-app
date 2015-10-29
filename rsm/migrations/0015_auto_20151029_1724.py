# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0014_auto_20151029_1414'),
    ]

    operations = [
        migrations.AddField(
            model_name='system',
            name='primary_output_display_name',
            field=models.TextField(default=b'Response value'),
        ),
        migrations.AlterField(
            model_name='input',
            name='plot_lower_bound',
            field=models.FloatField(default=0.0, help_text=b'Plots must be generated to show the true solution. What is the lower used in these plots for this variable? (Leave as zero for categorical variables.)'),
        ),
        migrations.AlterField(
            model_name='input',
            name='plot_upper_bound',
            field=models.FloatField(default=0.0, help_text=b'Plots must be generated to show the true solution. What is the upper used in these plots for this variable? (Leave as zero for categorical variables.)'),
        ),
    ]
