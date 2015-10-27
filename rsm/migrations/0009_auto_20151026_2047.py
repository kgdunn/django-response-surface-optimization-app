# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0008_input_units_as_prefix'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='input',
            name='units',
        ),
        migrations.RemoveField(
            model_name='input',
            name='units_as_prefix',
        ),
        migrations.AddField(
            model_name='input',
            name='units_prefix',
            field=models.CharField(help_text=b'The prefix for the units of the input (can be blank)', max_length=100, null=True, blank=True),
        ),
        migrations.AddField(
            model_name='input',
            name='units_suffix',
            field=models.CharField(help_text=b'The suffix for the units of the input (can be blank)', max_length=100, null=True, blank=True),
        ),
    ]
