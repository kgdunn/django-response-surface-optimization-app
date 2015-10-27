# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0011_auto_20151027_1526'),
    ]

    operations = [
        migrations.AlterField(
            model_name='input',
            name='n_decimals',
            field=models.PositiveSmallIntegerField(default=b'-1', help_text=b'The number of decimals to show in the numeric representation.', verbose_name=b'Number of decimals'),
        ),
    ]
