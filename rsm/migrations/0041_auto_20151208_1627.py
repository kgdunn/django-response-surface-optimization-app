# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0040_personsystem_solution_data'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='system',
            name='max_seconds_to_solve',
        ),
        migrations.AddField(
            model_name='system',
            name='max_seconds_before_solution',
            field=models.PositiveIntegerField(default=2147483647, help_text=b'Max seconds to wait before showing the solution. 43200=30 days, as an example.'),
        ),
    ]
