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

from rsm.models import PersonSystem
from rsm.views import update_leaderboard_score


persysts = PersonSystem.objects.all()

for persyst in persysts:
    update_leaderboard_score(persyst)