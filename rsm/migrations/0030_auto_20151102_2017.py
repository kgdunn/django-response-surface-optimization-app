# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0029_auto_20151102_1938'),
    ]

    operations = [
        migrations.AlterField(
            model_name='token',
            name='next_URI',
            field=models.CharField(default=b'', max_length=50, blank=True),
        ),
    ]
