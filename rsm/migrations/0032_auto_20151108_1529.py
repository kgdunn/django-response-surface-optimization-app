# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0031_auto_20151102_2217'),
    ]

    operations = [
        migrations.CreateModel(
            name='PlotHash',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('hash_value', models.CharField(default=b'--------------------------------', max_length=32, editable=False)),
                ('was_used', models.BooleanField(default=False)),
                ('time_last_used', models.DateTimeField(auto_now=True)),
                ('plot_HTML', models.TextField(default=b'', blank=True)),
                ('person', models.ForeignKey(to='rsm.Person')),
                ('system', models.ForeignKey(to='rsm.System')),
            ],
        ),
        migrations.RemoveField(
            model_name='token',
            name='plot_HTML',
        ),
    ]
