# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0007_auto_20151026_2040'),
    ]

    operations = [
        migrations.AddField(
            model_name='input',
            name='units_as_prefix',
            field=models.BooleanField(default=False, help_text=b'The default (False/left unchecked) means that units are shown in the suffix.'),
        ),
    ]
