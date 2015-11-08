# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0032_auto_20151108_1529'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='plothash',
            name='was_used',
        ),
    ]
