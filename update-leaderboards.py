#!/usr/bin/python

import os
import django


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import rsmproject.settings
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "rsmproject.settings"
)
django.setup()

from rsm.models import PersonSystem, Person
from rsm.views import update_leaderboard_score


persysts = PersonSystem.objects.all().order_by('system')

for persyst in persysts:
    update_leaderboard_score(persyst, note='KD: Adding regularity penalty.')

# Slugify the display names
people = Person.objects.all()
for person in people:
    person.save()