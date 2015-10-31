# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0022_auto_20151030_2207'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='experiment',
            name='is_valid',
        ),
        migrations.AddField(
            model_name='experiment',
            name='delete_by',
            field=models.DateTimeField(default=datetime.datetime(2015, 10, 31, 15, 26, 58, 962601, tzinfo=utc), verbose_name=b'Delete the experiment at this time if not validated.'),
        ),
        migrations.AddField(
            model_name='experiment',
            name='is_validated',
            field=models.BooleanField(default=False, help_text=b'False: indicates the Person has not validated their choice by signing in again.'),
        ),
        migrations.AddField(
            model_name='person',
            name='is_validated',
            field=models.BooleanField(default=False, help_text=b'Will be auto-validated once user has clicked on their email link.'),
        ),
    ]
