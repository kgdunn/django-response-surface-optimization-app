# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0003_auto_20151026_1629'),
    ]

    operations = [
        migrations.AlterField(
            model_name='system',
            name='noise_standard_deviation',
            field=models.FloatField(default=0, verbose_name=b'Standard deviation of normally distributed noise to add. Both normally and uniformly distributed noise will be added, if specified as non-zero values here.'),
        ),
        migrations.AlterField(
            model_name='system',
            name='simulation_timeout',
            field=models.PositiveSmallIntegerField(default=5, verbose_name=b'Seconds that may elapse before simulation is killed.'),
        ),
    ]
