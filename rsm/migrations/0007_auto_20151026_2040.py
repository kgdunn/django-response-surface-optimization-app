# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0006_auto_20151026_2039'),
    ]

    operations = [
        migrations.AlterField(
            model_name='input',
            name='lower_bound',
            field=models.FloatField(help_text=b'If supplied, will ensure the user does not enter a value below this.', null=True, blank=True),
        ),
    ]
