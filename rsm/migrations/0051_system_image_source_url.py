# -*- coding: utf-8 -*-
# Generated by Django 1.9.2 on 2016-03-07 19:19
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0050_system_min_experiments_allowed'),
    ]

    operations = [
        migrations.AddField(
            model_name='system',
            name='image_source_URL',
            field=models.CharField(default=b'', max_length=500),
        ),
    ]