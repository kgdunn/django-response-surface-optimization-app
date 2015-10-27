# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0005_auto_20151026_1941'),
    ]

    operations = [
        migrations.AlterField(
            model_name='input',
            name='upper_bound',
            field=models.FloatField(help_text=b'If supplied, will ensure the user does not enter a value above this.', null=True, blank=True),
        ),
    ]
