# -*- coding: utf-8 -*-
# Generated by Django 1.9.2 on 2016-02-28 21:49
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0044_auto_20160228_2148'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='personsystem',
            name='offsets',
        ),
        migrations.AddField(
            model_name='personsystem',
            name='offset_y',
            field=models.TextField(blank=True, default=b'', verbose_name=b'Offsets for system output'),
        ),
    ]
