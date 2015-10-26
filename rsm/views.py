# Dependancies (other than a basic Python + Django):
# pip install -U subprocess32

from django.shortcuts import get_object_or_404, render
from django.http import Http404, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.conf import settings as DJANGO_SETTINGS

import sys
import time
import math
import json
import decimal
import datetime
import logging.handlers


logger = logging.getLogger('RSMLogger')
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler(DJANGO_SETTINGS.LOG_FILENAME,
                                          maxBytes=2000000, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.debug('A new call to the views.py file')


if sys.version_info < (3, 2, 0):
    import subprocess32 as subprocess
else:
    import subprocess

from . import models

class RSMException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class WrongInputError(RSMException):
    """ Raised when a non-numeric input is provided."""
    pass

class OutOfBoundsInputError(RSMException):
    """ Raised when an input is outside the bounds."""
    pass

def run_simulation(system, simvalues):
    """Runs simulation of the required ``System``, with timeout. Returns a
    result no matter what, even if it is the default result (failure state).
    """
    # If Python < 3.x, then we require the non-builtin library ``subprocess32``
    start_time = time.clock()

    code = "\nimport numpy as np\n" + system.source
    code_call = """\n\nprint(simulate("""
    for key, value in simvalues.iteritems():
        code_call = code_call + "{0}={1}, ".format(key, value)

    code_call = code_call + "))"
    code = code + code_call

    #if


    command = r'python -c"{0}"'.format(code)
    proc = subprocess.Popen(command,
                            shell=True,
                            bufsize=-1,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    try:
        std_out, std_err = proc.communicate(None, timeout=1)

    except subprocess.TimeoutExpired:
        std_out = 'should_never_be_used_as_output'
        std_err = 'Timeout'
        logger.warn('Simulation of "{0}" timed out'.format(system))


    duration = time.clock() - start_time
    if std_err:
        # Typically: -987654321.0
        result = json.dumps({'output': system.default_error_output})
    else:
        result = std_out


        #the addition of noise to the primary output
        #    * runs any post-processing function specified in the model (e.g. clamping
        #      to certain minimum and maximum bounds)

    return (result, duration)

def process_simulation_input(values, inputs):
    """Cleans up the inputs from the web-based (typically human-readable) form,
    into numeric format expected by the simulation.

    Also ensure that the values are between the lower and upper bounds, if
    those were provided, for each input.
    """
    # NOTE: all inputs must be converted to floating point, to avoid any
    # discrepancy with integer division.
    out = {}
    try:
        for item in inputs:
            out[item.slug] = float(values[item.slug])
    except ValueError:
        raise WrongInputError(('Input "{0}" could not be converted to a '
                               'numeric value.').format(item.display_name))

    # Success! Now check the bounds.
    for item in inputs:
        if item.lower_bound is not None:
            if out[item.slug] < item.lower_bound:
                raise OutOfBoundsInputError(('Input "{0}" is below its lower '
                            'bound. It should be greater than or equal to {1}'
                            '.').format(item.display_name, item.lower_bound))

        if item.upper_bound is not None:
            if out[item.slug] > item.upper_bound:
                raise OutOfBoundsInputError(('Input "{0}" is above its upper '
                            'bound. It should be less than or equal to {1}'
                            '.').format(item.display_name, item.upper_bound))

        if math.isnan(out[item.slug]):
            raise OutOfBoundsInputError('Input "{0}" may not be "NaN".'.format(
                item.display_name))

        if math.isinf(out[item.slug]):
            raise OutOfBoundsInputError(('Input "{0}" may not be "-Inf" or '
                                         '"+Inf".').format(item.display_name))

    # End of checking all the inputs
    return out

def process_simulation_output(result, duration, next_run):
    """Cleans simulation output JSON, parses it, and returns it to be saved in
    the ``Results`` objects.

    The output processing includes:

    * adding the time-delay before results are displayed to the user.
    """
    pass

def show_all_systems(request):
    """
    Returns all the systems available to simulate at the user's current level.
    """
    system_list = models.System.objects.all()
    context = {'system_list': system_list}
    return render(request, 'rsm/root.html', context)


def process_experiment(request, short_name_slug):
    """ Processes the user's requested experiment; runs it, if it is valid.

    This is the POST handling of the System's webpage.
    """
    system = get_object_or_404(models.System, slug=short_name_slug)
    values = {}
    try:
        inputs = models.Input.objects.filter(system=system)
        for item in inputs:
            values[item.slug] = request.POST[item.slug]

        values_numeric = process_simulation_input(values, inputs)

    except (WrongInputError, OutOfBoundsInputError) as err:
        # Redisplay the experiment input form

        context = {'system': system,
                   'input_set': models.Input.objects.filter(system=system),
                   'error_message': ("You didn't properly enter some of "
                                     "the experimental input(s): "
                                     "{0}").format(err.value),
                   'values': values}
        return render(request, 'rsm/system-detail.html', context)
    else:

        # Success in checking the inputs. Create an input object for the user,
        # and run the experiment
        next_run = create_experiment_for_user(request, system, values_numeric)


        # TODO: try-except path goes here to intercept time-limited experiments

        # Clean-up the inputs by dropping any disallowed characters from the
        # function inputs:
        values_simulation = values_numeric.copy()
        for key in values_simulation.keys():
            value = values_simulation.pop(key)
            key = key.replace('-', '')
            values_simulation[key] = value

        result, duration = run_simulation(system, values_simulation)


        # Store the simulation results
        run_complete = process_simulation_output(result,
                                                 duration,
                                                 next_run)

        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.a
        return HttpResponseRedirect(reverse('rsmapp:show_one_system',
                                            args=(system.slug,)))

def show_one_system(request, short_name_slug):
    """ Returns the details of one system that has been selected, including
    the leaderboard (generated by another function)"""

    if request.POST:
        return process_experiment(request, short_name_slug)

    # If it was not a POST request, but a GET request...
    try:
        system = models.System.objects.get(slug=short_name_slug)
    except models.System.DoesNotExist:
        raise Http404(("Tried to find a system to optimize! But that system "
                       "does not exist."))

    fetch_leaderboard_results()


    # If the user is not logged in, show the input form, but it is disabled.
    # The user has to sign in with an email, and create a display name to
    # enter in experimental results. Come back to this part later.

    context = {'system': system,
               'input_set': models.Input.objects.filter(system=system)}
    return render(request, 'rsm/system-detail.html', context)


def inputs_to_JSON(inputs):
    """Converts the numeric inputs to JSON, after cleaning. This allows logging
    and storing them for all users.
    """
    return json.dumps(inputs)

def create_experiment_for_user(request, system, values_numeric, person=None):
    """Create the input for the given user"""
    # TODO: Currently the "Person" is None. This will be added in the future,
    #       typing the input object to a specific user.

    # TODO: check that the user is allowed to create a new input at this point
    #       in time. There might be a time limitation in place still.
    #       If so, offer to store the input and run it at the first possible
    #       occasion.

    next_run = models.Experiment(person=models.Person.objects.get(id=1),
               token=models.Token.objects.get(id=1),
               system=system,
               inputs=inputs_to_JSON(values_numeric),
               is_valid=False,
               time_to_solve=-500,
               earliest_to_show=datetime.datetime(datetime.MAXYEAR, 12, 31,
                                                  23,59,59))
    return next_run



def fetch_leaderboard_results(system=None):
    """ Returns the leaderboard for the current system.
    """
    pass




