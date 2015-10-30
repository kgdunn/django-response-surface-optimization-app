# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0021_token_plot_created'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='token',
            name='plot_created',
        ),
        migrations.AddField(
            model_name='token',
            name='plot_HTML',
            field=models.TextField(default=b''),
        ),
    ]
