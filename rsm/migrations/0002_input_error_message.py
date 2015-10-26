# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='input',
            name='error_message',
            field=models.CharField(help_text=b'Any error message text that should be shown during input validation.', max_length=200, blank=True),
        ),
    ]
