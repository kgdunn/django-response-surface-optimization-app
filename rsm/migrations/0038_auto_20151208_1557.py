# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import datetime
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0037_auto_20151208_1507'),
    ]

    operations = [
        migrations.AddField(
            model_name='personsystem',
            name='started_on',
            field=models.DateTimeField(default=datetime.datetime(2015, 12, 8, 15, 57, 22, 237258, tzinfo=utc), auto_now_add=True),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='system',
            name='max_seconds_to_solve',
            field=models.PositiveIntegerField(default=2147483647, help_text=b'Max seconds to wait before showing the solution. 43200=30 days, as an example.'),
        ),
        migrations.AlterField(
            model_name='personsystem',
            name='frozen',
            field=models.BooleanField(default=False, help_text=b'If true, prevents any further additions to this system.'),
        ),
        migrations.AlterField(
            model_name='personsystem',
            name='offsets',
            field=models.TextField(default=b'', verbose_name=b'Offsets for each system input'),
        ),
        migrations.AlterField(
            model_name='personsystem',
            name='rotation',
            field=models.PositiveSmallIntegerField(default=0, help_text=b'Rotation around axis for this system'),
        ),
    ]
