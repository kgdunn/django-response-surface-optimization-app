# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0035_auto_20151115_0830'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='experiment',
            name='is_validated',
        ),
    ]
