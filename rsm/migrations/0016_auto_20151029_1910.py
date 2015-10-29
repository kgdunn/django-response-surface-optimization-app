# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0015_auto_20151029_1724'),
    ]

    operations = [
        migrations.RenameField(
            model_name='system',
            old_name='primary_output_display_name',
            new_name='primary_output_display_name_with_units',
        ),
    ]
