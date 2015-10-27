# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0012_auto_20151027_1528'),
    ]

    operations = [
        migrations.AlterField(
            model_name='input',
            name='n_decimals',
            field=models.PositiveSmallIntegerField(default=0, help_text=b'The number of decimals to show in the numeric representation.', verbose_name=b'Number of decimals'),
        ),
    ]
