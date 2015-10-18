# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Person',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(max_length=200)),
                ('level', models.SmallIntegerField(default=0, verbose_name=b'Skill level of the user')),
                ('email', models.EmailField(max_length=254)),
            ],
        ),
        migrations.CreateModel(
            name='Result',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('is_valid', models.BooleanField()),
                ('time_to_solve', models.FloatField(default=0.0, verbose_name=b'Time to solve model')),
                ('earliest_to_show', models.DateTimeField(verbose_name=(b"Don't show the ", b'result before this point in time'))),
                ('main_result', models.FloatField(default=-987654321.0, verbose_name=b'Primary numeric output')),
                ('other_outputs', models.TextField(null=True, verbose_name=b'Other outputs produced, including string messages, in JSON format, as defined by the ``System``.', blank=True)),
                ('person', models.ForeignKey(to='rsm.Person')),
            ],
        ),
        migrations.CreateModel(
            name='System',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('full_name', models.CharField(max_length=250)),
                ('description', models.TextField(unique=True, verbose_name=b'A description of what this system does, and hints on the objective of the optimization.')),
                ('level', models.PositiveSmallIntegerField(default=0, verbose_name=b'Skill level required by user')),
                ('source', models.TextField(unique=True, verbose_name=b'Python source code that will be executed. Called function with name ``simulate(...)`` must exist')),
                ('simulation_timeout', models.PositiveSmallIntegerField(default=0, verbose_name=b'Seconds that may elapse before simulation is killed.')),
                ('default_error_output', models.FloatField(default=-987654321.0, verbose_name=b'The default value assigned when the simulation fails.')),
                ('n_inputs', models.PositiveSmallIntegerField(verbose_name=b'Number of model inputs')),
                ('n_outputs', models.PositiveSmallIntegerField(verbose_name=b'Number of model outputs')),
                ('output_json', models.TextField(verbose_name=b'Comma-separated list of modeloutput names; the first onemust be "result"')),
                ('delay_result', models.IntegerField(verbose_name=b'Number of seconds before the result may be shown to users.')),
                ('known_peak_inputs', models.TextField(verbose_name=b'JSON structure giving the input(s) known to produce a maximum', blank=True)),
                ('noise_standard_deviation', models.FloatField(default=0, verbose_name=b'Amount of normally distributed noise to add.')),
            ],
        ),
        migrations.CreateModel(
            name='Tag',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('short_name', models.CharField(max_length=50)),
                ('description', models.CharField(max_length=150)),
            ],
        ),
        migrations.CreateModel(
            name='Token',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('was_used', models.BooleanField(default=False)),
                ('time_used', models.DateTimeField()),
                ('ip_address', models.GenericIPAddressField()),
                ('person', models.ForeignKey(to='rsm.Person')),
                ('system', models.ForeignKey(to='rsm.System')),
            ],
        ),
        migrations.AddField(
            model_name='system',
            name='tags',
            field=models.ManyToManyField(to='rsm.Tag'),
        ),
        migrations.AddField(
            model_name='result',
            name='system',
            field=models.ForeignKey(to='rsm.System'),
        ),
        migrations.AddField(
            model_name='result',
            name='token',
            field=models.ForeignKey(to='rsm.Token'),
        ),
    ]
