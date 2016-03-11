# -*- coding: utf-8 -*-
# Generated by Django 1.9.2 on 2016-03-11 15:59
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rsm', '0056_remove_experiment_delete_by'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='message_while_waiting',
            field=models.CharField(default=b'', help_text=b"Message to display while 'running' the experiment.'", max_length=510),
        ),
    ]
