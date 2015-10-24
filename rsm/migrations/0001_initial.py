# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Input',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('display_name', models.CharField(max_length=200)),
                ('slug', models.SlugField()),
                ('ntype', models.CharField(default=b'CON', max_length=3, verbose_name=b'The numeric type of the input variable.', choices=[(b'CON', b'Continuous'), (b'CAT', b'Categorical')])),
                ('level_numeric_mapping', models.TextField(help_text=b'For example: {"water": "-1", "vinegar": "+1"} would map the "water" level to -1 and "vinegar" level to +1 in the simulation. Leave blank for continuous variables.', verbose_name=b'Specify UNIQUE names for each numeric level of a categorical variable; JSON format.', blank=True)),
                ('lower_bound', models.FloatField(help_text=b'If supplied, will ensure the user does not enter a value below this.', blank=True)),
                ('upper_bound', models.FloatField(help_text=b'If supplied, will ensure the user does not enter a value above this.', blank=True)),
                ('default_value', models.FloatField(help_text=b'The default used, e.g. in a multidimensional (>3) plot.')),
                ('units', models.CharField(help_text=b'The units of the input', max_length=100)),
            ],
        ),
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
                ('earliest_to_show', models.DateTimeField(verbose_name=b"Don't show the result before this point in time")),
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
                ('slug', models.SlugField()),
                ('description', models.TextField(default=b'The aim of this ...', unique=True, verbose_name=b'A description of what this system does, and hints on the objective of the optimization.')),
                ('level', models.PositiveSmallIntegerField(default=0, verbose_name=b'Skill level required by user')),
                ('source', models.TextField(default='def simulate(A, B, ):\n    # Code here', unique=True, verbose_name=b'Python source code that will be executed. A function with the name ``simulate(...)`` must exist. The NumPy library is available as ``np``.')),
                ('simulation_timeout', models.PositiveSmallIntegerField(default=0, verbose_name=b'Seconds that may elapse before simulation is killed.')),
                ('default_error_output', models.FloatField(default=-987654321.0, verbose_name=b'The default value assigned when the simulation fails.')),
                ('n_inputs', models.PositiveSmallIntegerField(default=1, verbose_name=b'Number of model inputs')),
                ('n_outputs', models.PositiveSmallIntegerField(default=1, verbose_name=b'Number of model outputs')),
                ('output_json', models.TextField(default=b'result', verbose_name=b'Comma-separated list of model output names; the first one must be "result"')),
                ('delay_result', models.IntegerField(default=0, verbose_name=b'Number of seconds before the result may be shown to users.')),
                ('known_peak_inputs', models.TextField(verbose_name=b'JSON structure giving the input(s) known to produce a maximum', blank=True)),
                ('noise_standard_deviation', models.FloatField(default=0, verbose_name=b'Amount of normally distributed noise to add. Both normally and uniformly distributed noise will be added, if specified as non-zero values here.')),
                ('noise_uniform_multiplier', models.FloatField(default=0, verbose_name=b"Multiplier for uniformally distributed noise: y = mx + c; this is for multiplier 'm'.")),
                ('noise_uniform_offset', models.FloatField(default=0, verbose_name=b"Offset for uniformally distributed noise: y = mx + c; this is for offset value 'c'.")),
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
        migrations.AddField(
            model_name='input',
            name='system',
            field=models.ForeignKey(to='rsm.System'),
        ),
    ]
