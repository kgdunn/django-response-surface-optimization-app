# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0023_auto_20151031_1526'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='delete_by',
            field=models.DateTimeField(default=datetime.datetime(2015, 10, 31, 15, 27, 2, 452987, tzinfo=utc), verbose_name=b'Delete the experiment at this time if not validated.'),
        ),
    ]
