# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0018_auto_20151030_2147'),
    ]

    operations = [
        migrations.AddField(
            model_name='system',
            name='cost_per_experiment',
            field=models.FloatField(default=10.0, help_text=b'Dollar cost per run'),
        ),
        migrations.AddField(
            model_name='system',
            name='max_experiments_allowed',
            field=models.PositiveIntegerField(default=100),
        ),
    ]
