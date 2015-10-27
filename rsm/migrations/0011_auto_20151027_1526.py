# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0010_input_formatting_string'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='input',
            name='formatting_string',
        ),
        migrations.AddField(
            model_name='input',
            name='n_decimals',
            field=models.SmallIntegerField(default=b'-1', help_text=b'The number of decimals to show in the numeric representation.'),
        ),
    ]
