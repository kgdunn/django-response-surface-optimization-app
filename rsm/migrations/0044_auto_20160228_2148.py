# -*- coding: utf-8 -*-
# Generated by Django 1.9.2 on 2016-02-28 21:48
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0043_system_image_description'),
    ]

    operations = [
        migrations.AlterField(
            model_name='system',
            name='image_description',
            field=models.ImageField(null=True, upload_to=b'rsm/static/rsm/'),
        ),
    ]
