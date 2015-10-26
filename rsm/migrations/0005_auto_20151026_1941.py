# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0004_auto_20151026_1941'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='system',
            name='noise_standard_deviation',
        ),
        migrations.RemoveField(
            model_name='system',
            name='noise_uniform_multiplier',
        ),
        migrations.RemoveField(
            model_name='system',
            name='noise_uniform_offset',
        ),
    ]
