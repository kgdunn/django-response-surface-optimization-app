# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0002_input_error_message'),
    ]

    operations = [
        migrations.CreateModel(
            name='Experiment',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('is_valid', models.BooleanField()),
                ('time_to_solve', models.FloatField(default=0.0, verbose_name=b'Time to solve model')),
                ('earliest_to_show', models.DateTimeField(verbose_name=b"Don't show the result before this point in time")),
                ('inputs', models.TextField(verbose_name=b'The system inputs logged in JSON format')),
                ('main_result', models.FloatField(default=-987654321.0, verbose_name=b'Primary numeric output')),
                ('other_outputs', models.TextField(null=True, verbose_name=b'Other outputs produced, including string messages, in JSON format, as defined by the ``System``.', blank=True)),
            ],
        ),
        migrations.RemoveField(
            model_name='result',
            name='person',
        ),
        migrations.RemoveField(
            model_name='result',
            name='system',
        ),
        migrations.RemoveField(
            model_name='result',
            name='token',
        ),
        migrations.AlterField(
            model_name='person',
            name='name',
            field=models.CharField(max_length=200, verbose_name=b'Leaderboard name'),
        ),
        migrations.DeleteModel(
            name='Result',
        ),
        migrations.AddField(
            model_name='experiment',
            name='person',
            field=models.ForeignKey(to='rsm.Person'),
        ),
        migrations.AddField(
            model_name='experiment',
            name='system',
            field=models.ForeignKey(to='rsm.System'),
        ),
        migrations.AddField(
            model_name='experiment',
            name='token',
            field=models.ForeignKey(to='rsm.Token'),
        ),
    ]
