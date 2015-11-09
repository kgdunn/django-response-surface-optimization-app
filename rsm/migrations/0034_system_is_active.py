# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0033_remove_plothash_was_used'),
    ]

    operations = [
        migrations.AddField(
            model_name='system',
            name='is_active',
            field=models.BooleanField(default=False, help_text=b'If False, then this system will not be usable.'),
        ),
    ]
