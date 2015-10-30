# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0019_auto_20151030_2152'),
    ]

    operations = [
        migrations.AlterField(
            model_name='token',
            name='ip_address',
            field=models.GenericIPAddressField(default=b'127.0.0.1'),
        ),
        migrations.AlterField(
            model_name='token',
            name='time_used',
            field=models.DateTimeField(auto_now=True),
        ),
    ]
