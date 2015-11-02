# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0030_auto_20151102_2017'),
    ]

    operations = [
        migrations.AlterField(
            model_name='token',
            name='plot_HTML',
            field=models.TextField(default=b'', blank=True),
        ),
    ]
