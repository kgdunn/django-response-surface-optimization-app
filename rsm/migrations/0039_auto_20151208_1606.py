# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0038_auto_20151208_1557'),
    ]

    operations = [
        migrations.AlterField(
            model_name='personsystem',
            name='offsets',
            field=models.TextField(default=b'', verbose_name=b'Offsets for each system input', blank=True),
        ),
    ]
