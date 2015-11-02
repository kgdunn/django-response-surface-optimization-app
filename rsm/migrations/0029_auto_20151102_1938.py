# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0028_auto_20151102_1932'),
    ]

    operations = [
        migrations.AddField(
            model_name='token',
            name='experiment',
            field=models.ForeignKey(blank=True, to='rsm.Experiment', null=True),
        ),
        migrations.AddField(
            model_name='token',
            name='next_URI',
            field=models.CharField(default=b'', max_length=50),
        ),
    ]
