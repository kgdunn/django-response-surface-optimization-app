# -*- coding: utf-8 -*-
# Generated by Django 1.9.2 on 2016-03-07 20:10
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0051_system_image_source_url'),
    ]

    operations = [
        migrations.AlterField(
            model_name='personsystem',
            name='offset_y',
            field=models.FloatField(blank=True, default=0.0, verbose_name=b'Offset for system output'),
        ),
    ]