# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0026_auto_20151031_1622'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='delete_by',
            field=models.DateTimeField(auto_now=True, verbose_name=b'Delete the experiment at this time if not validated.'),
        ),
        migrations.AlterField(
            model_name='person',
            name='email',
            field=models.EmailField(unique=True, max_length=254),
        ),
    ]
