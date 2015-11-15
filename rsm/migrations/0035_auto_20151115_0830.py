# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0034_system_is_active'),
    ]

    operations = [
        migrations.AlterField(
            model_name='token',
            name='system',
            field=models.ForeignKey(blank=True, to='rsm.System', null=True),
        ),
    ]
