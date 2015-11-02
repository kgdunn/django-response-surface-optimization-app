# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0027_auto_20151101_1232'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='token',
            name='ip_address',
        ),
        migrations.AlterField(
            model_name='person',
            name='level',
            field=models.SmallIntegerField(default=1, verbose_name=b'Skill level of the user'),
        ),
    ]
