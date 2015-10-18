# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='system',
            name='noise_uniform_multiplier',
            field=models.FloatField(default=0, verbose_name=b"Multiplier for uniformally distributed noise: y = mx + c; this is for multiplier 'm'."),
        ),
        migrations.AddField(
            model_name='system',
            name='noise_uniform_offset',
            field=models.FloatField(default=0, verbose_name=b"Offset for uniformally distributed noise: y = mx + c; this is for offset value 'c'."),
        ),
        migrations.AlterField(
            model_name='system',
            name='output_json',
            field=models.TextField(verbose_name=b'Comma-separated list of model output names; the first onemust be "result"'),
        ),
    ]
