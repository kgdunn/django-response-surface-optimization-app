# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0024_auto_20151031_1527'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='experiment',
            name='token',
        ),
        migrations.AddField(
            model_name='experiment',
            name='hash_value',
            field=models.CharField(default=b'--------------------------------', max_length=32, editable=False),
        ),
        migrations.AlterField(
            model_name='experiment',
            name='delete_by',
            field=models.DateTimeField(default=datetime.datetime(2015, 10, 31, 16, 6, 23, 757679, tzinfo=utc), verbose_name=b'Delete the experiment at this time if not validated.'),
        ),
        migrations.AlterField(
            model_name='experiment',
            name='is_validated',
            field=models.BooleanField(default=False, help_text=b'False: indicates the Person has not validated their choice by signing in (again).'),
        ),
    ]
