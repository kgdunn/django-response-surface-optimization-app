# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0039_auto_20151208_1606'),
    ]

    operations = [
        migrations.AddField(
            model_name='personsystem',
            name='solution_data',
            field=models.TextField(help_text=b'In JSON format', blank=True),
        ),
    ]
