# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0036_remove_experiment_is_validated'),
    ]

    operations = [
        migrations.CreateModel(
            name='PersonSystem',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('rotation', models.PositiveSmallIntegerField(help_text=b'Rotation around axis for this system')),
                ('offsets', models.TextField(verbose_name=b'Offsets for each system input')),
                ('completed_date', models.DateTimeField()),
                ('show_solution_as_of', models.DateTimeField()),
                ('frozen', models.BooleanField(default=False)),
                ('person', models.ForeignKey(to='rsm.Person')),
                ('system', models.ForeignKey(to='rsm.System')),
            ],
        ),
        migrations.AlterField(
            model_name='token',
            name='person',
            field=models.ForeignKey(blank=True, to='rsm.Person', null=True),
        ),
    ]
