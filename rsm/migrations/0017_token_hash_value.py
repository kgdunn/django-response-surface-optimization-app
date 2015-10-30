# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0016_auto_20151029_1910'),
    ]

    operations = [
        migrations.AddField(
            model_name='token',
            name='hash_value',
            field=models.CharField(default=b'--------------------------------', max_length=32, editable=False),
        ),
    ]
