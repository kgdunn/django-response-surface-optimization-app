# -*- coding: utf-8 -*-
# Generated by Django 1.9.2 on 2016-03-10 12:34
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0053_auto_20160310_1205'),
    ]

    operations = [
        migrations.AddField(
            model_name='system',
            name='known_optimum_response',
            field=models.FloatField(default=-9999999999),
        ),
    ]
