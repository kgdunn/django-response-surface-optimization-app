# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0009_auto_20151026_2047'),
    ]

    operations = [
        migrations.AddField(
            model_name='input',
            name='formatting_string',
            field=models.CharField(default=b'{}', help_text=b'A formatting string that indicates how the numeric value is converted to string (HTML).', max_length=50),
        ),
    ]
