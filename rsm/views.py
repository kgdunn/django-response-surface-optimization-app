# Dependancies (other than a basic Python + Django):
# pip install -U subprocess32

from django.shortcuts import get_object_or_404, render
from django.http import Http404, HttpResponseRedirect
from django.core.urlresolvers import reverse

import sys
import time
import math

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

def run_simulation():
    """Runs simulation of the required ``System``, with timeout. Returns a
    result no matter what, even if it is the default result (failure state).
    """
    # https://docs.python.org/2/library/functions.html#compile
    #https://stackoverflow.com/questions/2983963/run-a-external-program-with-specified-max-running-time

    # https://stackoverflow.com/questions/1191374/subprocess-with-timeout

    start_time = time.clock()
    code = r"""
    def simulate(A):
        coded = (A - 135)/15.0
        y = round(coded * 15 - 2.4 * coded * coded + 93, 1)
        return '{{\"output\": {0}}}'.format(*(y,))
    """

    code = "\nimport numpy as np\n" + code + "\n\nprint(simulate(A=227))"
    command = r'python -c"{0}"'.format(code)
    proc = subprocess.Popen(command, shell=True, bufsize=-1,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        std_out, std_err = proc.communicate(None, timeout=1)

    except subprocess.TimeoutExpired:
        std_out = 0.0
        std_err = 'Timeout'

    duration = time.clock() - start_time

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

def process_simulation_output(outputs):
    """Cleans simulation output JSON, parses it, and returns it to be saved in
    the ``Results`` objects.

    The output processing includes:
    * the addition of noise to the primary output
    * runs any post-processing function specified in the model (e.g. clamping
      to certain minimum and maximum bounds)
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


def run_experiment(request, short_name_slug):
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
        create_input_for_user(request, system, values_numeric)
        sim_out = run_simulation(system, values_numeric)
        process_simulation_output(sim_out)

        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('rsmapp:show_one_system',
                                            args=(system.slug,)))

def show_one_system(request, short_name_slug):
    """ Returns the details of one system that has been selected, including
    the leaderboard (generated by another function)"""

    if request.POST:
        return run_experiment(request, short_name_slug)

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


def create_input_for_user(request, system, values_numeric):
    """Create the input for the given user"""

    pass



def fetch_leaderboard_results(system=None):
    """ Returns the leaderboard for the current system.
    """
    pass




