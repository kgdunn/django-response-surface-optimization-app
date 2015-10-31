# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0025_auto_20151031_1606'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='was_successful',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='experiment',
            name='delete_by',
            field=models.DateTimeField(default=datetime.datetime(2015, 10, 31, 16, 22, 1, 103345, tzinfo=utc), verbose_name=b'Delete the experiment at this time if not validated.'),
        ),
    ]
