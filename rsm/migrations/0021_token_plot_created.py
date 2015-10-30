# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0020_auto_20151030_2159'),
    ]

    operations = [
        migrations.AddField(
            model_name='token',
            name='plot_created',
            field=models.BooleanField(default=False),
        ),
    ]
