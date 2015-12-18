# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0041_auto_20151208_1627'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='plothash',
            name='person',
        ),
        migrations.RemoveField(
            model_name='plothash',
            name='system',
        ),
        migrations.AddField(
            model_name='personsystem',
            name='plot_HTML',
            field=models.TextField(default=b'', blank=True),
        ),
        migrations.AddField(
            model_name='personsystem',
            name='plot_hash',
            field=models.CharField(default=b'--------------------------------', max_length=32, editable=False),
        ),
        migrations.AlterField(
            model_name='personsystem',
            name='solution_data',
            field=models.TextField(default=b'', help_text=b'In JSON format', blank=True),
        ),
        migrations.AlterField(
            model_name='system',
            name='source',
            field=models.TextField(default='def simulate(**inputs):\n  # Code here', unique=True, verbose_name=b'Python source code that will be executed. A function with the name ``simulate(...)`` must exist. The NumPy library is available as ``np``.'),
        ),
        migrations.DeleteModel(
            name='PlotHash',
        ),
    ]
