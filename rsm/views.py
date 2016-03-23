# Dependancies:
# pip install -U django
# pip install -U numpy
# pip install -U subprocess32
# pip install -U pillow   <--- not yet


from django.shortcuts import get_object_or_404, render
from django.http import (Http404, HttpResponseRedirect, HttpResponse,
                         HttpResponseForbidden)
from django.core.urlresolvers import reverse, NoReverseMatch
from django.conf import settings as DJANGO_SETTINGS
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.db.utils import IntegrityError
from django.template.loader import render_to_string
from django.utils.timezone import utc

import os
import sys
import time
import math
import json
import decimal
import random
import hashlib
from datetime import date, datetime, timedelta, MAXYEAR
from datetime import time as dt_time

from smtplib import SMTPException
from collections import defaultdict, namedtuple
import logging
import numpy as np

#os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'
#import matplotlib
#matplotlib.use( 'Agg' )
#from matplotlib.figure import Figure


# Some settings for this app:
TOKEN_LENGTH = 5

logger = logging.getLogger(__name__)
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

class MissingInputError(RSMException):
    """ One of the inputs is not supplied (radio/catagorical only)"""

class WrongInputError(RSMException):
    """ Raised when a non-numeric input is provided."""
    pass

class OutOfBoundsInputError(RSMException):
    """ Raised when an input is outside the bounds."""
    pass

class Rotation(object):
    def __init__(self, slidescale, dim=None, rotation_matrix=''):
        """Will construct a rotation object it will
        randomly create a rotation matrix for the appropriate dimension.

        The slidescale matrix should also be supplied, it is the (ndim x 2)
        matrix. Column 1 = lower bounds for the ``ndim`` inputs, and Column 2
        is the upper bound for each input.
        """
        self.dim = dim
        self.slidescale = slidescale
        if rotation_matrix == '':
            if self.dim == 1:
                self.rotmat = np.array([[np.cos(0),]])
            if self.dim == 2:
                theta = np.random.randint(0, 360) * np.pi/180.0
                logger.debug('A new random rotation: {0}'.format(theta*180/np.pi))
                self.rotmat = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta),  np.cos(theta)]])
            if self.dim >= 3:
                assert(False)

        elif rotation_matrix:
            self.rotmat = np.array(json.loads(rotation_matrix))
            self.dim = self.rotmat.shape[0]

    def get_rotation_string(self):
        """
        Returns the rotation matrix for this class as a string.
        An object of this class can then subsequently be reconstructed as:

        object = Rotation(slidescale=str_slidescale,
                          rotation_matrix=str_rotation_matrix)
        """
        if self.rotmat is None:
            return ''
        else:
            return json.dumps(self.rotmat.tolist())

    def forward_rotate(self, data):
        """ Applies the forward rotation.

        Multiplies the (ndim x ndim) rotation matrix with the (ndim x N) data
        matrix, where N is the number of datapoints being individually rotated.
        The output is an (ndim x N) matrix of rotated points.
        """
        if isinstance(self.rotmat, basestring) and self.rotmat == '':
            return data
        offset =  np.sum(self.slidescale, axis=1) / 2.0  # midpoint
        offset = offset.reshape(self.dim, 1)
        scale = np.diff(self.slidescale, axis=1) / 6.0  # from high to low

        data_sc = (data - offset) / scale
        rotated = np.dot(self.rotmat, data_sc)
        return rotated * scale + offset

    def inverse_rotate(self, data):
        """ Applies the forward rotation.

        Multiplies the (ndim x ndim) inverse rotation matrix with the
        (ndim x N) data matrix, where N is the number of datapoints being
        individually rotated.
        The output is an (ndim x N) matrix of inverse rotated points.
        """
        return np.dot(np.linalg.inv(self.rotmat), data)

def run_simulation(system, simvalues, show_solution=False):
    """Runs simulation of the required ``System``, with timeout. Returns a
    result no matter what, even if it is the default result (failure state).

    Note: when ``show_solution=True`` then two things which are unusual happen:
        1/ a new function is appended and executed first that takes all inputs
           and if they are NumPy arrays represented as lists, then are converted
           back to NumPy arrays first, before the ``simulate(...)`` function is
           called.

        2/ the ``post_process(...)`` function is not executed
    """
    logger.debug('Running simultion for : {0}'.format(system.slug))
    start_time = time.clock()

    code = "\nimport numpy as np\n"

    if show_solution:
        code += "def convert_inputs(**kwargs):\n"
        code += "\tout = {}\n"
        code += "\tfor key, value in kwargs.iteritems():\n"
        code += "\t\tout[key] = np.array(value)\n"
        code += "\treturn out\n\n"
        code += "def convert_outputs(**kwargs):\n"
        code += "\tout = {}\n"
        code += "\tfor key, value in kwargs.iteritems():\n"
        code += "\t\tif isinstance(value, np.ndarray):\n"
        code += "\t\t\tout[key] = value.tolist()\n"
        code += "\t\telse:\n"
        code += "\t\t\tout[key] = np.array(value)\n"
        code += "\treturn out\n\n"
        code += system.source
    else:
        code += system.source


    code_call = '\n\nout = simulate('
    if show_solution:
        code_call += '**convert_inputs('

    for key, value in simvalues.iteritems():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        code_call = code_call + "{0}={1}, ".format(key, value)

    if show_solution:
        code_call += "))\n"
    else:
        code_call += ")\n"

    code += code_call

    if not(show_solution):
        if (r"post_process(" in system.source):
            code += "print(post_process(out))"
        else:
            code += "print(out)"
    else:
        code += "out = convert_outputs(**out)\nprint(out)"

    command = r'python -c"{0}"'.format(code)

    # If Python < 3.x, then we require the non-builtin library ``subprocess32``
    proc = subprocess.Popen(command,
                            shell=True,
                            bufsize=-1,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    try:
        std_out, std_err = proc.communicate(None,
                                            timeout=system.simulation_timeout)
    except subprocess.TimeoutExpired:
        std_out = 'should_never_be_used_as_output'
        std_err = 'Timeout'
        logger.warn('Simulation of "{0}" timed out'.format(system))

    duration = time.clock() - start_time
    if std_err:
        # Typically: -987654321.0
        result = json.dumps({'output': system.default_error_output})
        logger.warning('Simulation FAILED: {0}. Error={1}'.format(
                            system.full_name, std_err))
    else:
        result = std_out

    return (result, duration)

def process_simulation_inputs_templates(inputs, request=None, force_GET=False):
    """ Cleans up the inputs so they are rendered appropriately in the Django
    templates.
    The categorical variable's numeric levels are split out, and modified
    into an actual Python dict (not a Django database object)
    """
    categoricals = {}
    for item in inputs:
        # Continuous items need no processing at the moment
        # Categorical items
        if item.ntype == 'CAT':
            dict_string = json.loads(item.level_numeric_mapping)
            categoricals[item.slug] = dict_string
            if force_GET:
                if item.slug in request.POST:
                    categoricals[item.slug][request.POST[item.slug]] = '__checked__'


    return (inputs, categoricals)

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
            if item.ntype == 'CON':
                out[item.slug] = float(values[item.slug])
            elif item.ntype == 'CAT':
                types = json.loads(item.level_numeric_mapping)
                out[item.slug] = types[values[item.slug]]

    except (ValueError, KeyError):  # KeyError: for the categorical variables
        raise WrongInputError(('"{0}" could not be converted to a '
                               'numeric value.').format(item.display_name))

    # Success! Now check the bounds.
    for item in inputs:
        if item.lower_bound is not None:
            if out[item.slug] < item.lower_bound:
                raise OutOfBoundsInputError(('"{0}" is below its lower '
                            'bound. It should be greater than or equal to {1}'
                            '.').format(item.display_name, item.lower_bound))

        if item.upper_bound is not None:
            if out[item.slug] > item.upper_bound:
                raise OutOfBoundsInputError(('"{0}" is above its upper '
                            'bound. It should be less than or equal to {1}'
                            '.').format(item.display_name, item.upper_bound))

        if math.isnan(out[item.slug]):
            raise OutOfBoundsInputError('"{0}" may not be "NaN".'.format(
                item.display_name))

        if math.isinf(out[item.slug]):
            raise OutOfBoundsInputError(('"{0}" may not be "-Inf" or '
                                         '"+Inf".').format(item.display_name))

    # End of checking all the inputs
    return out

def process_simulation_output(result, next_run, system, is_baseline):
    """Cleans simulation output text, parses it, and returns it to be saved in
    the ``Experiment`` objects. The output is returned, but not saved here (that
    is to be done elsewhere).
    """
    result = result.replace('\n', '')
    result = result.replace("'", '"')
    result = json.loads(result)

    # Store the numeric result:
    next_run.main_result = result.pop('output')
    if next_run.main_result == system.default_error_output:
        next_run.was_successful = False
    else:
        next_run.was_successful = True

    # Are there any other outputs produced?
    if result:
        next_run.other_outputs = json.dumps(result)

    if is_baseline:
        next_run.earliest_to_show = datetime.now().replace(tzinfo=utc)
    else:
        # Adding the time-delay before results are displayed to the user.
        next_run.earliest_to_show = datetime.now().replace(tzinfo=utc) + \
                        timedelta(seconds=system.delay_result)

    return next_run

def create_token_send_email_check_success(person, system_slug, system):
    """ Used during signing in a new user, or an existing user. A token to
    is created, and an email is sent.
    If the email succeeds, then we return with success, else, we indicate
    failure to the calling function.
    """
    # Create a token for the new user
    hash_value = generate_random_token(TOKEN_LENGTH)
    try:
        next_URI = reverse('rsmapp:show_one_system', args=(system_slug,))
        system = models.System.objects.get(is_active=True, slug=system_slug)
    except NoReverseMatch:
        next_URI = ''

    # Send them an email
    failed = send_suitable_email(person, hash_value)

    if failed: # SMTPlib cannot send an email
        return False
    else:
        token = models.Token(person=person,
                             system=system,
                             hash_value=hash_value,
                             experiment=None,
                             next_URI=DJANGO_SETTINGS.WEBSITE_BASE_URI+next_URI)

        token.next_URI = next_URI.strip(DJANGO_SETTINGS.WEBSITE_BASE_URI)
        return token

def popup_sign_in(request):
    """POST-only sign-in via the website. """

    # NOTE: this uses the fact that the URLs are /system/abc
    # We aim to find the system so we can redirect the person clicking on
    # an email.
    referrer = request.META.get('HTTP_REFERER', '/').split('/')
    try:
        system_slug = referrer[referrer.index('system')+1]
        system = models.System.objects.get(is_active=True, slug=system_slug)
    except (ValueError, IndexError, models.System.DoesNotExist):
        system_slug=''
        system = None

    if 'emailaddress' not in request.POST:
        return HttpResponse("Unauthorized access", status=401)

    # Process the sign-in
    # 1. Check if email address is valid based on a regular expression check.
    try:
        email = request.POST.get('emailaddress', '').strip()
        validate_email(email)
    except ValidationError:
        return HttpResponse("Invalid email address. Try again please.",
                            status=406)

    # 2. Is the user signed in already? Return back (essentially do nothing).
    # TODO: handle this case still. For now, just go through with the email
    #       again (but this is prone to abuse). Why go through? For the case
    #       when a user signs in, now the token is used. But if they reuse that
    #       token to sign in, but the session here is still active, they can
    #       potentially not sign in, until they clear their cookies.
    #if request.session.get('person_id', False):
    #    return HttpResponse("You are already signed in.", status=200)

    # 3A: a brand new user, or
    # 3B: a returning user that has cleared cookies/not been present for a while
    try:
        # Testing for 3A or 3B
        person = models.Person.objects.get(email=email)

        # Must be case 3B. If prior failure, then it is case 3A (see below).
        token = create_token_send_email_check_success(person, system_slug,
                                                      system)
        if token:
            token.save()
            return HttpResponse(("<i>Welcome back!</i> Please check your email,"
                     " and click on the link that we emailed you."), status=200)
        else:
            return HttpResponse(("An email could not be sent to you. Please "
                                 "ensure your email address is correct."),
                                status=404)

    except models.Person.DoesNotExist:
        # Case 3A: Create totally new user. At this point we are sure the user
        #          has never been validated on our site before.
        #          But the email address they provided might still be faulty.
        person = models.Person(is_validated=False,
                               display_name='Anonymous',
                               email=email)
        person.save()
        person.display_name = person.display_name + str(person.id)

        token = create_token_send_email_check_success(person, system_slug,
                                                      system)
        if token:
            person.save()
            token.person = person  # must overwrite the prior "unsaved" person
            token.save()
            return HttpResponse(("An account has been created for you, but must"
                                 " be actived. Please check your email and "
                                 "click on the link that we emailed you."),
                                status=200)
        else:
            # ``token`` will automatically be forgotten when this function
            # returns here. Perfect!
            person.delete()
            return HttpResponse(("An email could NOT be sent to you. Please "
                "ensure your email address is valid."), status=404)


def show_all_systems(request):
    """
    Returns all the systems available to simulate at the user's current level.
    """
    system_list = models.System.objects.filter(is_active=True).order_by('level')
    person, enabled_status = get_person_info(request)

    solved_list = np.zeros(len(system_list)).tolist()
    if enabled_status:

        for idx, system in enumerate(system_list):
            persyst = models.PersonSystem.objects.filter(person=person,
                                                        system=system)
            # i.e. the person has not yet attempted this system: therefore
            if len(persyst) == 0:
                solved_list[idx] = 0
            else:
                solved_list[idx] = persyst[0].is_solved

    systems = [{'system': t[0], 'solved': t[1]} for t in zip(system_list,
                                                             solved_list)]
    context = {'system_list': system_list,
               'person': person,
               'enabled': enabled_status,
               'systems': systems,
              }
    return render(request, 'rsm/show-all-systems.html', context)

def process_experiment(request, short_name_slug):
    """ Processes the user's requested experiment; runs it, if it is valid.

    This is the POST handling of the System's webpage.
    """
    system = get_object_or_404(models.System, slug=short_name_slug,
                               is_active=True)
    values = {}
    values_checked = {}
    try:
        # We do all our checks here, and if any fail an Exception is raised.
        # There are several checks: data entry, etc.

        # NB: Read the user values first before doing any checking on them.
        inputs = models.Input.objects.filter(system=system).order_by('slug')

        for item in inputs:
            try:
                values[item.slug] = request.POST[item.slug]
            except KeyError:
                raise MissingInputError(('Input "{0}" was not specified.'.
                                         format(item.display_name)))

        # We've got all the inputs now; so validate them.
        values_checked.update(process_simulation_input(values, inputs))

        # TODO.v2: try-except path here to intercept time-limited experiments
        # if a new run is not valid, then raise an exception.
        # Technically, a regular user from the website cannot run a run
        # before the elapsed time, since the form is not displayed. But,
        # you might want to add code here anyway to ensure that.

    except (WrongInputError, OutOfBoundsInputError, MissingInputError) as err:

        logger.warn('User error raised: {0}. Context:{1}'.format(err.value,
                                                                 str(values)))
        # Redisplay the experiment input form if any invalid data enty.
        # Redirect back to ``show_one_system()`` so you don't repeat code.
        extend_dict = {'error_message': ("You didn't properly enter some of "
                                         "the required input(s): "
                                         "{0}").format(err.value),
                       'prior_values': values}
        return show_one_system(request, short_name_slug, force_GET=True,
                       extend_dict=extend_dict)

    # Success! at this point all inputs have been checked.
    # Create an input object for the user, and run the experiment.
    # The ``Person`` object will be found from ``request``
    next_run, values_checked, persyst = create_experiment_object(request,
                                                                 system,
                                                                 values_checked)

    next_run = execute_experiment_object(next_run, persyst, values_checked)

    # Return an HttpResponseRedirect after dealing with POST. Prevents data
    #from being posted twice if a user hits the Back button.
    return HttpResponseRedirect(reverse('rsmapp:show_one_system',
                                        args=(system.slug,)))

def get_person_info(request):
    """
    Returns a tuple: the ``person`` object, and where or not that
    user has been activated/enabled. The latter is determined by their
    username == '__Anonymous__'
    """
    # Get the user. Assume, to start, that it is an anonymous/unknown user.
    person = models.Person.objects.get(display_name='__Anonymous__',
                                       email='anonymous@learnche.org')
    if request.session.get('person_id', False):
        try:
            person = models.Person.objects.get(id=request.session['person_id'])
        except models.Person.DoesNotExist:
            pass

    # enabled_status = True if the person has signed in AND validated themself.
    enabled_status = (person.is_validated == True) or \
                                        (person.display_name != '__Anonymous__')

    return person, enabled_status


def reset_one_system(request, short_name_slug):
    """ Resets the current system for the logged in user.
    """
    # Get the current ``person``. If ``enabled_status=False`` would indicate
    # it is an anonymous, or unvalidated person.
    person, enabled_status = get_person_info(request)

    if request.GET and not(force_GET) or not(enabled_status):
        return process_experiment(request, short_name_slug)

    # Get the relevant input objects for this system
    # If it was not a POST request, but a (possibly forced) GET request...
    try:
        system = models.System.objects.get(slug=short_name_slug,
                                           is_active=True)
    except models.System.DoesNotExist:
        return process_experiment(request, short_name_slug)

    # Have there been any prior experiments for this person?
    prior_expts = models.Experiment.objects.filter(system=system, person=person)
    persysts = models.PersonSystem.objects.filter(system=system, person=person)
    for expt in prior_expts:
        expt.delete()
    for persyst in persysts: # should always be one of these
        persyst.delete()

    return HttpResponseRedirect(reverse('rsmapp:show_one_system',
                                        args=(system.slug,)))

def show_solution_one_system(request, short_name_slug):
    """ Sets things to show the solution for the current system
    """
    # Get the current ``person``. If ``enabled_status=False`` would indicate
    # it is an anonymous, or unvalidated person.
    person, enabled_status = get_person_info(request)

    if request.GET and not(force_GET) or not(enabled_status):
        return process_experiment(request, short_name_slug)

    # Get the relevant input objects for this system
    # If it was not a POST request, but a (possibly forced) GET request...
    try:
        system = models.System.objects.get(slug=short_name_slug,
                                           is_active=True)
    except models.System.DoesNotExist:
        return process_experiment(request, short_name_slug)

    # Have there been any prior experiments for this person?
    persysts = models.PersonSystem.objects.filter(system=system, person=person)
    if len(persysts) < 1:
        return process_experiment(request, short_name_slug)

    persyst = persysts[0]
    persyst.completed_date = datetime.now().replace(tzinfo=utc)
    persyst.show_solution_as_of = datetime.now().replace(tzinfo=utc)
    persyst.save()

    logger.debug('Set things up to show the solution; now going to render it.')

    return HttpResponseRedirect(reverse('rsmapp:show_one_system',
                                        args=(system.slug,)))



def show_one_system(request, short_name_slug, force_GET=False, extend_dict={}):
    """ Returns the details of one system that has been selected, including
    the leaderboard (generated by another function).

    ``force_GET`` and ``extend_dict`` are only used when the POST version,
    ``process_experiment(...)`` fails due to bad user input.
    """

    if request.POST and not(force_GET):

        # Ensure a person is signed in.
        if request.session.get('person_id', 0) == 0:
            return show_one_system(request, short_name_slug, force_GET=True,
                        extend_dict={'message': ('You must sign in before '
                                                 'running any experiments.')})
        else:
            return process_experiment(request, short_name_slug)

    # If it was not a POST request, but a (possibly forced) GET request...
    try:
        system = models.System.objects.get(slug=short_name_slug,
                                           is_active=True)
    except models.System.DoesNotExist:
        raise Http404(("Tried to find a system to optimize! But that system "
                       "does not exist."))


    # Get the current ``person``
    person, enabled_status = get_person_info(request)

    logger.debug("Showing a system for person {0} ".format(person))

    # Get the relevant input objects for this system
    input_set = models.Input.objects.filter(system=system).order_by('slug')

    extra_information = ''
    show_solution = False

    # Use a fake object: this is only for anonymous users, since they are
    # not allowed to interact with the systems until signed in.
    persyst_fake = namedtuple('FakeObject', ['plot_HTML',])
    persyst = persyst_fake(plot_HTML='')

    if enabled_status:
        # If enabled, it allows the user to interact with this system.

        # Initiate the ``PersonSystem`` for this combination only once
        persysts = models.PersonSystem.objects.filter(system=system,
                                                      person=person)

        # If this is zero, it is because it is the first time the person has
        # visited this system
        if persysts.count() == 0:
            logger.debug("First visit to {0} for person {1} ".format(system,
                                                                     person))
            future = datetime(MAXYEAR, 12, 31, 23, 59, 59).\
                replace(tzinfo=utc)
            solution_date = datetime.now() + \
                               timedelta(0, system.max_seconds_before_solution)
            solution_date = solution_date.replace(tzinfo=utc)

            persyst = models.PersonSystem(person=person, system=system,
                                          completed_date=future, frozen=False,
                                          show_solution_as_of=solution_date)
            persyst.save()
        else:
            persyst = persysts[0]
            logger.debug("Returning visitor {1} to system {0}.".format(system,
                                                                       person))


        # Have there been any prior experiments for this person?
        if models.Experiment.objects.filter(system=system, person=person,
                                            was_successful=True).count() == 0:

            # If not: Create a baseline run for the person at the default values
            default_values = {}

            # Ensure that input_set is in alphabetical order of slug
            for inputi in input_set:
                default_values[inputi.slug] = inputi.default_value

            baseline, default_values, persyst = create_experiment_object(\
                                               request, system, default_values)

            lower_bound, upper_bound = json.loads(system.offset_y_range)
            persyst.offset_y = np.random.randint(lower_bound, upper_bound)
            baseline = execute_experiment_object(baseline,
                                                 persyst,
                                                 default_values,
                                                 is_baseline=True)

        # Should we show the solution? Let's check:
        #if datetime.now().replace(tzinfo=utc) > \
        #                       persyst.show_solution_as_of.replace(tzinfo=utc):
        if persyst.is_solved:
            show_solution = True
            logger.debug("Solution being shown on account of date/time")

        n_expts = models.Experiment.objects.filter(system=persyst.system,
                                                   person=persyst.person,
                            was_successful=True).count()
        if n_expts >= system.max_experiments_allowed:
            if persyst.completed_date  > datetime.now().replace(tzinfo=utc):
                persyst.completed_date = datetime.now().replace(tzinfo=utc)
                persyst.save()

            show_solution = True
            extra_information = ('You have reached the maximum number of '
                                 'experiments allowed. The solution will be '
                                 'automatically displayed below.')
            logger.debug("Solution being shown on account n_expts >= max")

        if show_solution:
            # Have we generated the solution before? If so, don't do it again.
            # But if not, generate it once, and then save it (it is expensive).
            if persyst.frozen == False:
                logger.debug('Solution has not been generated; about to...')
                persyst = generate_solution(persyst)
                logger.debug('Solution generated; about to freeze it.')

                # Freeze it once the solution has been generated.
                persyst.frozen = True
                persyst.save()


        # Only get this for signed in users
        plot_raw_data = get_plot_and_data_HTML(persyst, input_set,
                                               show_solution)

    else:
        plot_raw_data = [ ]

    # if-else-end: if enabled_status

    leads = fetch_leaderboard_results_one_system(system=system, person=person)

    input_set, categoricals = process_simulation_inputs_templates(input_set,
                                                                  request,
                                                                  force_GET)

    context = {'system': system,
               'input_set': input_set,
               'leads' : leads,
               'person': person,
               'enabled': enabled_status,
               'extra_information': extra_information,
               'plot_html': persyst.plot_HTML,
               'data_html': plot_raw_data,
               'show_solution': show_solution,
               'number_remaining': (system.max_experiments_allowed - \
                            len(plot_raw_data) ),
               'budget_remaining': (system.max_experiments_allowed - \
                            len(plot_raw_data) )*system.cost_per_experiment,
               }
    context.update(extend_dict)   # used for the ``force_GET`` case when the
                                  # user has POSTed prior invalid data.
    context['categoricals'] = categoricals
    return render(request, 'rsm/system-detail.html', context)

def adequate_username(request, username):
    """Checks if the username is adequate: not taken, not offensive, and 4 or
    more characters."""
    unique = models.Person.objects.filter(display_name=username).count() == 0
    length = len(username) >= 4
    # TODO.v2: check for offensive names
    return length*unique

def validate_user(request, hashvalue):
    """ The new/returning user has been sent an email to sign in.
    Recall their token, mark them as validated, sign them in, run the experiment
    they had intended, and redirect them to the next URL associated with their
    token.

    If it is a new user, make them select a Leaderboard name first.
    """
    logger.info('Locating validation token {0}'.format(hashvalue))
    token = get_object_or_404(models.Token, hash_value=hashvalue)

    message = ''
    request_new_token = False

    if token.was_used:
        # Prevents a token from being re-used.
        message = ('That validation key has been already used. Please request '
                   'another by clicking on the "Sign-in" button on the home page.')
        request_new_token = True

    if request.POST:
        if adequate_username(request, request.POST['rsm_username'].strip()):
            username = request.POST['rsm_username'].strip()
            logger.info('NEW USER: {0}'.format(username))

            send_logged_email(subject="RSM: New user: {0}".format(username),
                              message="New RSM user: {0}".format(username),
                              to_address_list=[DJANGO_SETTINGS.ADMINS[0][1],])
            token.person.display_name = username
            token.person.is_validated = True
            token.person.save()
            token.was_used = True
            token.save()
            return sign_in_user(request, hashvalue, renderit=False)
        else:
            # Too short, or already used. Try again ...
            token.was_used = False
            token.save()
            return HttpResponse("That username is too short, or already exists",
                                status=406)
    else:
        message = message or ('Thank you for validating your email address. '
                              'Now please select a user name for yourself...')
        context = {'hashvalue': hashvalue,
                   'message': message,
                   'suggestions': create_fake_usernames(10),
                   'person': token.person,
                   'enabled': True,
                   'hide_sign_in': True
                  }
        # Force the user to request a new token, as that one has been used.
        if request_new_token:
            context.pop('person')
            context.pop('suggestions')
            context['enabled'] = False

        return render(request, 'rsm/choose-new-leaderboard-name.html', context)

def sign_in_user(request, hashvalue, renderit=True):
    """ User is sign-in with the unique hashcode sent to them,
        These steps are used once the user has successfully been validated,
        or if sign-in is successful.

        A user is considered signed-in if "request.session['person_id']" returns
        a valid ``person.id`` (used to look up their object in the DB)
        """

    logger.debug('Attempting sign-in with token {0}'.format(hashvalue))
    token = get_object_or_404(models.Token, hash_value=hashvalue)
    token.was_used = True
    token.save()
    request.session['person_id'] = token.person.id
    logger.info('RETURNING USER: {0}'.format(token.person.display_name))

    if token.system:
        next_uri = reverse('rsmapp:show_one_system', args=(token.system.slug,))
    else:
        next_uri = reverse('rsmapp:show_all_systems')

    if renderit:
        if token.system:
            next_uri = reverse('rsmapp:show_one_system',
                                                     args=(token.system.slug,))
            content = show_one_system(request, token.system.slug,
                                                                force_GET=True)
        else:
            next_uri = reverse('rsmapp:show_all_systems')
            content = show_all_systems(request)

        # Now return that content
        return HttpResponseRedirect(next_uri, content=content)
    else:
        # This case is used when the user is just getting a redirect (no
        # rendered content)
        return HttpResponse(next_uri, status=200)

def send_suitable_email(person, hash_val):
    """ Sends a validation email, and logs the email message. """

    if person.is_validated:
        sign_in_URI = '{0}/sign-in/{1}'.format(DJANGO_SETTINGS.WEBSITE_BASE_URI,
                                        hash_val)
        ctx_dict = {'sign_in_URI': sign_in_URI,
                    'username': person.display_name}
        message = render_to_string('rsm/email_sign_in_code.txt',
                                   ctx_dict)
        subject = ("Unique code to sign-into the Response Surface "
                   "Methods website.")
        to_address_list = [person.email.strip('\n'), ]

    else:
        # New users / unvalidated user
        check_URI = '{0}/validate/{1}'.format(DJANGO_SETTINGS.WEBSITE_BASE_URI,
                                             hash_val)
        ctx_dict = {'validation_URI': check_URI}
        message = render_to_string('rsm/email_new_user_to_validate.txt',
                                   ctx_dict)

        subject = ("Confirm your email address for the Response Surface "
                   "Methods website!")
        to_address_list = [person.email.strip('\n'), ]


    # Use regular Python code to send the email in HTML format.
    message = message.replace('\n','\n<br>')
    return send_logged_email(subject, message, to_address_list)

def create_experiment_object(request, system, values, N_values=1):
    """Create the input for the given user. BUT, it does not save it. This is
    intentional.
    """
    # TODO: check that the user is allowed to create a new input at this point
    #       in time. There might be a time limitation in place still.
    #       If so, offer to store the input and run it at the first possible
    #       occasion.

    if request.session.get('person_id', False):
        person = models.Person.objects.get(id=request.session['person_id'])
    else:
        logger.error('Unlogged user attempted to create an experiment.')
        assert(False)

    values['_rot_'] = np.zeros((system.continuous_dimensionality(), N_values))
    values['_ss_'] = np.zeros((system.continuous_dimensionality(), 2))

    # Ensure that input_set is in alphabetical order of slug
    input_set = models.Input.objects.filter(system=system).order_by('slug')
    persysts = models.PersonSystem.objects.filter(system=system, person=person)
    persyst = persysts[0]

    dim = 0
    for inputi in input_set:
        if inputi.ntype == 'CON':
            # This is the value that is about to be rotated
            values['_rot_'][dim] = values[inputi.slug]
            values['_ss_'][dim, :] = [inputi.plot_lower_bound,
                                      inputi.plot_upper_bound]
            values[dim] = inputi.slug
            values['__' + inputi.slug + '__'] = values[inputi.slug]
            dim += 1

    # Apply the rotation here: ensure that continuous values are applied
    # in alphabetical order:
    rot_obj = Rotation(dim=system.continuous_dimensionality(),
                       slidescale=values['_ss_'],
                       rotation_matrix=persyst.rotation.encode())
    persyst.rotation = rot_obj.get_rotation_string()
    persyst.save()
    rotated = rot_obj.forward_rotate(values['_rot_'])

    idx = 0
    for inputi in input_set:
        if inputi.ntype == 'CON':
            values[values[idx]] = rotated[idx][0]
            values.pop(idx)
            idx += 1

    # Clean-up "values" before calling the simulation.
    values.pop('_rot_')
    values.pop('_ss_')

    next_run = models.Experiment(person=person,
                                 system=system,
                                 inputs=inputs_to_JSON(values),
                                 time_to_solve=-500,
                                 earliest_to_show=
                      datetime(MAXYEAR, 12, 31, 23, 59, 59).replace(tzinfo=utc))
    return next_run, values, persyst


def execute_experiment_object(expt_obj, persyst, values, is_baseline=False):
    """Typically called after ``create_experiment_object`` once all inputs
    have been cleaned and checked.

    Also updates the leaderboard score for this user.
    """
    # Clean-up the inputs by dropping any disallowed characters from the
    # function inputs:
    values_simulation = values.copy()
    for key in values_simulation.keys():
        value = values_simulation.pop(key)
        key = key.replace('-', '')
        values_simulation[key] = value

    result, duration = run_simulation(persyst.system, values_simulation)
    expt_obj.time_to_solve = duration

    # Store the simulation results and return the experimental object
    expt_obj = process_simulation_output(result, expt_obj, persyst.system,
                                         is_baseline)

    # Add biasing that is specific for this user
    if expt_obj.was_successful:
        expt_obj.main_result = expt_obj.main_result + persyst.offset_y

    # Swap the actual (rotated) value used in the simulation with
    # the user required value. So that the user sees the display
    # as they entered.
    input_set = models.Input.objects.filter(system=persyst.system).order_by('slug')
    for inputi in input_set:
        if inputi.ntype == 'CON':
            actual = values[inputi.slug]
            user_value = values['__' + inputi.slug + '__']
            values[inputi.slug] = user_value
            values['__' + inputi.slug + '__'] = actual

    expt_obj.inputs = inputs_to_JSON(values)
    expt_obj.save()

    # Update the leaderboard value
    update_leaderboard_score(persyst)

    return expt_obj

def generate_solution(persyst):
    """Generates a solution for a given Person/System combination. This is
    expensive process, so save the results. """
    system = persyst.system

    # TODO v3. Handle categorical variables here.
    input_set = models.Input.objects.filter(system=system).order_by('slug')

    values = {}
    RESOLUTION = 50
    data, hash_value = get_person_experimental_data(persyst, input_set)

    idx = 0
    for inputi in input_set:
        input_name = inputi.slug.replace('-', '')


        # Deal with continuous and categorical inputs differently.
        if inputi.ntype == 'CON':
            # Generate the contour plot in the range that the user worked in.
            sub_data = data[inputi.slug]
            range_min, range_max, delta = plotting_defaults(sub_data,
                clamps=[inputi.plot_lower_bound, inputi.plot_upper_bound],
                force_clamps=True)

            values[input_name] = np.linspace(start=range_min,
                                                      stop=range_max,
                                                      num=RESOLUTION,
                                                      endpoint=True)
            # Get the data necessary to construct the meshgrid
            if idx == 0 and len(input_set) >= 2:
                x = values[input_name]
                xname = input_name
            if idx == 1 and len(input_set) >= 2:
                y = values[input_name]
                yname = input_name

            idx += 1

        elif inputi.ntype == 'CAT':
            pass

    # Special processing for 2D systems
    if len(input_set) >= 2:
        values[xname], values[yname] = np.meshgrid(x, y)

        # Apply the necessary rotation here
        values['_rot_'] = np.zeros((system.continuous_dimensionality(),
                                    RESOLUTION*RESOLUTION))
        values['_ss_'] = np.zeros((system.continuous_dimensionality(), 2))

        dim = 0
        for inputi in input_set:
            if inputi.ntype == 'CON':
                values['_rot_'][dim] = values[inputi.slug.replace('-', '')]\
                                                  .reshape(RESOLUTION*RESOLUTION)
                values['_ss_'][dim, :] = [inputi.plot_lower_bound,
                                          inputi.plot_upper_bound]
                values[dim] = inputi.slug.replace('-', '')
                dim += 1

        # Apply the rotation here: ensure that continuous values are applied
        # in alphabetical order:
        rot_obj = Rotation(dim=system.continuous_dimensionality(),
                           slidescale=values['_ss_'],
                           rotation_matrix=persyst.rotation.encode())
        rotated = rot_obj.forward_rotate(values['_rot_'])

        idx = 0
        for inputi in input_set:
            if inputi.ntype == 'CON':
                values[values[idx]] = rotated[idx].reshape(RESOLUTION,
                                                           RESOLUTION)
                values.pop(idx)
                idx += 1

        # Clean-up "values" before calling the simulation.
        values.pop('_rot_')
        values.pop('_ss_')

    # Ignore post_process(): which is where the noise is added.
    result, duration = run_simulation(system, values,
                                      show_solution=True)

    # Finally, store the simulation results and return the experimental object
    class FakeClass(): pass
    results = FakeClass()
    results = process_simulation_output(result, results, system,
                                        is_baseline=False)
    solution_data = {}

    if len(input_set) >= 2:
        X, Y = np.meshgrid(x, y)
        for idx, inputi in enumerate(input_set):

            # We must use the shortened variable names, and we use the unrotated
            # data point
            input_name = inputi.slug

            if inputi.ntype == 'CON':
                if idx == 0:
                    values[xname] = X
                if idx == 1:
                    values[yname] = Y

    solution_data['inputs'] = serialize_numeric_dict(values)
    if results.was_successful:
        bias_soln = (np.array(results.main_result) + persyst.offset_y).tolist()
        solution_data['outputs'] = bias_soln
    else:
        logger.error('Error generating solution for {0}'.format(persyst))
        assert(False)

    persyst.solution_data = json.dumps(solution_data, allow_nan=True)
    return persyst

def fetch_leaderboard_results_one_system(system=None, person=None):
    """ Returns the leaderboard for the current system.
    """
    persysts = models.PersonSystem.objects.filter(system=system)

    MAX_NUMBER = 10
    leads = []
    found_you = False
    for idx, persyst in enumerate(persysts):
        you = 0
        if person == persyst.person:
            you = 1
            found_you = True

        # Leaderboard tuple:
        # 1. Score
        # 2. The person's display name
        # 3. A boolean indicating if it is them (logged in user) or not
        leads.append([persyst.get_score(), persyst.person.display_name, you, 0])

    # Sort by the first field (the score, from highest to lowest)
    leads.sort(reverse=True)
    for idx, item in enumerate(leads):
        item[3] = idx + 1
        leads[idx] = item

    if not(found_you) or (len(leads) <= MAX_NUMBER):
        return leads[0:MAX_NUMBER]
    else:
        where_are_you = -1
        for idx, item in enumerate(leads):
            if item[2] == 1:
                where_are_you = idx

        # Sadly, this user will be off the list. Ensure that they are added
        if (where_are_you+1) >= (MAX_NUMBER):
            this_user = leads[where_are_you-1:where_are_you+2]
            leads = leads[0:MAX_NUMBER-len(this_user)]
            leads.extend(this_user)
            return leads
        else:
            return leads[0:MAX_NUMBER]

def update_leaderboard_score(persyst):
        """Calculates and updates the leaderboard score and stores it for the
        current Person/System combination at the date/time that the result is to
        be revealed. This means that the calculate value may be for a future
        reveal time."""
        leaderboard = json.loads(persyst.leaderboard)
        input_set = models.Input.objects.filter(system=persyst.system).\
                                                       order_by('slug')

        expts, hash_value = get_person_experimental_data(persyst, input_set)

        responses = expts['_output_']
        now_update = [{}, expts['_datetime_'][-1].strftime("%Y-%m-%dT%H:%M:%S")]
        max_output = np.max(responses)

        # Use a 75/25    blend of the maximum output and the last response
        # that the user used. The last experiment should, when complete, be
        # run at the optimum, and then this weighted sum will have weights = 1.0
        user_peak = 0.75*max_output + 0.25*responses[-1]
        true_opt = persyst.system.known_optimum_response

        # Start with the closeness to the optimum. Don't forget to remove
        # the offset that has been artificially added. This should now
        # get you a number that is close to 0.0 if you are at the optimum.
        score = np.abs((true_opt - (user_peak-persyst.offset_y))/(true_opt ))
        score = (1.0 - score)*100.0
        now_update[0]['closeness'] = score

        # Now account for the minumum number of experiments, and the actual
        # number of experments. This reduces the current score. It has the
        # effect that users will see their score increase for the first few
        # experiments, even though they are not necessarily getting closer
        # to the optimum.
        run_penalty = np.power(np.abs(len(expts['_output_']) - \
                                persyst.system.min_experiments_allowed), 0.9)
        now_update[0]['run_penalty'] = run_penalty

        score = score - run_penalty
        now_update[0]['score'] = score
        leaderboard.append(now_update)
        persyst.leaderboard = json.dumps(leaderboard)
        persyst.save()

def get_person_experimental_data(persyst, input_set):
    """Gets the data for a person and returns it, together with a hash value
    that should/is unique up to that point.

    The experiments are returned in the order they were run.
    """
    data = defaultdict(list)

    # Retrieve prior experiments which were successful, for this system,
    # for the current logged in person. Note: ALSO experiments that are only
    # going to be revealed in the future are used here. So that the hash
    # is complete. It is critical that the experiments be ordered as shown,
    # since we sometime will use [-1] to refer to the last one.
    prior_expts = models.Experiment.objects.filter(system=persyst.system,
                                                   person=persyst.person,
                            was_successful=True).order_by('earliest_to_show')
    data_string = str(persyst.person) + ';' + str(persyst.system)
    for entry in prior_expts:
        inputs = json.loads(entry.inputs)
        data['_datetime_'].append(entry.earliest_to_show)
        for item in input_set:
            data[item.slug].append(inputs[item.slug])

            # Append these to the string:
            data_string += str(data[item.slug])

        # After processing all inputs, also process the response value:
        data['_output_'].append(entry.main_result)
        data_string += str(data['_datetime_'])

    # Append the outputs, and the solution data
    data_string += str(data['_output_'])
    data_string += persyst.solution_data

    if not(data['_output_']):
        hash_value = None
    else:
        hash_value = hashlib.md5(data_string).hexdigest()

        # If the hash has changed, then delete the old HTML
        if hash_value != persyst.plot_hash:
            persyst.plot_HTML = ''
        persyst.plot_hash = hash_value
        persyst.save()
    return data, hash_value

def plotting_defaults(vector, clamps=None, force_clamps=False):
    """ Finds suitable clamping ranges and a "dy" offset to place marker
    labels using heuristics.

    Independent of how plots are generated.

    If ``force_clamps`` is True, it ensures that the bounds at least include
    the ``clamps``. Note that if ``vector`` contains data outside the range
    of ``clamps`` that the range will exceed ``clamps``.
    """
    finite_clamps = True
    if clamps is None:
        clamps = [float('-inf'), float('+inf')]
        finite_clamps = False

    y_min = np.nanmax([np.nanmin(vector), clamps[0]])
    y_max = np.nanmin([np.nanmax(vector), clamps[1]])

    if force_clamps:
        if y_min > clamps[0]:
            y_min = clamps[0]
        if y_max < clamps[1]:
            y_max = clamps[1]

    y_range = y_max - y_min
    if y_range == 0.0 and finite_clamps:
        y_min, y_max = clamps
        y_range = y_max - y_min
    elif y_range == 0.0 and not(finite_clamps):
        y_range = 1.0

    y_range_min, y_range_max = y_min - 0.07*y_range, y_max + 0.07*y_range
    dy = 0.015*y_range
    return (y_range_min, y_range_max, dy)

def plot_wrapper(data, persyst, inputs, hash_value, show_solution=False):
    """Creates a plot of the data, and returns the HTML code to display the
    plot.
    Optionally shows the solution if ``show_solution`` is True.
    """
    def get_axis_label_name(input_item):
        """Returns an axis label for a particular input.

        Independent of how plots are generated.
        """
        if input_item.units_prefix and not(input_item.units_suffix):
            return '{0} [{1}]'.format(input_item.display_name,
                                      input_item.units_prefix)
        elif not(input_item.units_prefix) and input_item.units_suffix:
            return '{0} [{1}]'.format(input_item.display_name,
                                      input_item.units_suffix)
        elif input_item.units_prefix + input_item.units_suffix:
            return '{0} [{1} {2}]'.format(input_item.display_name,
                                          input_item.units_prefix,
                                          input_item.units_suffix)
        else:
            return input_item.display_name

    def add_data_labels_NOT_USED(axis, dims, xvalues, yvalues, zvalues=None,
                        dx=0, dy=0, dz=0, rotate=False):
        """Adds labels to an axis ``axis`` in 1, 2 or 3 dimensions.
        If ``rotate`` is True, then the labels are randomly rotated
        (placed) about the plotted point. The ``dx``, ``dy`` and ``dz`` are
        the base offsets from the coordinate values given in ``xvalues``,
        ``yvalues``, and ``zvalues`` respectively.
        """
        for idx, xvalue in enumerate(xvalues):
            if rotate:
                theta = np.random.uniform(0, 2*np.pi)
            else:
                theta = 0.0

            rdx = dx*np.cos(theta) - dy*np.sin(theta)
            rdy = dx*np.sin(theta) + dy*np.cos(theta)

            axis.text(xvalue+rdx,
                      y_data[idx]+rdy,
                      str(idx+1),
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=10,
                      family='serif')

    # 0. Create the empty figure
    # 1. Get  data to plot, and the numeric limits for each axis from the inputs
    # 2. Create the axes and set the axes names
    # 3. Create the gridlines
    # 4. Plot title
    # 5. Now add the actual data points

    # ...: Use a marker size proportional to objective function
    # ...: Add labels for each point

    # 7. Add a legend
    # 8. What to do when we mouse over the plot?
    # 9. Close off and render the plot

    logger.debug('Plot generation: part 0')

    # 0. Create the empty figure
    #=========================
    plot_HTML = """
    <style type="text/css">
	.axis path, .axis line {
		fill: none;
		stroke: #000;
		shape-rendering: crispEdges;
	}
	.tick{
		font: 12px sans-serif;
	}
	.grid .tick {
	    stroke: lightgrey;
	}
    .expt-results{
        display: inline-block;
        *display: inline;
        vertical-align: middle;
        zoom: 1;
    }

    </style>
    <div id="rsmchart" class="expt-results" ></div>
    <script type="text/javascript">"""

    plot_HTML += """
        // Global defaults
        var showlegend = false;
        var n_ticks_x = 8;
        var n_ticks_y = 8;
        var deltabuffer = 5; // small buffers away from axes
        var margin = {top:40, right:showlegend?120:50, bottom:40, left:50 };

        var chartDiv = document.getElementById("rsmchart");
        var svgcontainer = d3.select(chartDiv);

        function redraw_rsmchart(){
            svgcontainer.selectAll("*").remove();

            // Extract the width and height that was computed by CSS.
            // But clamp it to a maximum width of 600px. Ideally then the
            // table of results is side-by-side on a wide screen monitor.
            var outerwidth = Math.min(600, Math.max(600, chartDiv.clientWidth));
            var outerheight = 400;
            var width = outerwidth - margin.left - margin.right;
            var height = outerheight - margin.top - margin.bottom;

            // ``range``: the output scale mapped to SVG port dimensions
            var scalex = d3.scale.linear().range([0, width]);
            var scaley = d3.scale.linear().range([height, 0]);

        var svg = svgcontainer.append("svg")
            .attr("width", outerwidth)
            .attr("height", outerheight)
            .attr('class','rsm-figure')

            // Everything that will be added to the plot is now relative to this
            //transformed container.
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        // To help with layout: show the boundaries
        // Do not ordinarily display this.
        // svg.append("rect")
        //     .attr("width", width)
        //     .attr("height", height)
        //     .attr("opacity", 0.2);

        // Set the axes, as well as details on their ticks
        var xAxis = d3.svg.axis()
            .scale(scalex)
            .ticks(n_ticks_x)
            .tickSubdivide(true)
            .tickSize(6, 0);

        var yAxis = d3.svg.axis()
            .scale(scaley)
            .ticks(n_ticks_y)
            .tickSubdivide(true)
            .tickSize(6, 0);
        """

    # 1. Get the data to plot and the plot bounds
    #=========================
    logger.debug('Plot generation: part 1')
    responses = data['_output_']
    x_range_min, x_range_max, y_range_min, y_range_max = 0, 0, 0, 0
    if show_solution:
        soldata = json.loads(persyst.solution_data)

    if persyst.system.continuous_dimensionality() == 1:
        temp_x = x_data = data[inputs[0].slug]
        temp_y = y_data = responses

        if show_solution:
            x_soldata = soldata['inputs'][inputs[0].slug.replace('-', '')]
            y_soldata = soldata['outputs']
            temp_x = x_data[:] # important: make a copy!
            temp_x.extend(x_soldata)

            temp_y = y_data[:] # here as well
            temp_y.extend(y_soldata)



        x_range_min, x_range_max, dx = plotting_defaults(temp_x,
            clamps=[inputs[0].plot_lower_bound, inputs[0].plot_upper_bound])

        if len(responses) == 1:
            # Special case: when just initializing the system and the
            # only experiment is our baseline experiment.
            y_range_min, y_range_max, dy = plotting_defaults(responses,
                clamps=[responses[0]*0.9, responses[0]*1.10])
        else:
            y_range_min, y_range_max, dy = plotting_defaults(temp_y)

    elif persyst.system.continuous_dimensionality() == 2:
        # To ensure we always present the data in the same way
        inputs = inputs.filter(ntype='CON').order_by('slug')

        x_data = data[inputs[0].slug]
        y_data = data[inputs[1].slug]

        if show_solution:
            X = soldata['inputs'][inputs[0].slug.replace('-', '')]
            Y = soldata['inputs'][inputs[1].slug.replace('-', '')]
            temp_x = np.array(X).ravel().tolist()
            temp_x.extend(x_data)

            temp_y = np.array(Y).ravel().tolist()
            temp_y.extend(y_data)
        else:
            temp_x, temp_y = x_data, y_data

        x_range_min, x_range_max, dx = plotting_defaults(temp_x,
                clamps=[inputs[0].plot_lower_bound, inputs[0].plot_upper_bound])
        y_range_min, y_range_max, dy = plotting_defaults(temp_y,
                clamps=[inputs[1].plot_lower_bound, inputs[1].plot_upper_bound])



    elif persyst.system.continuous_dimensionality() >= 3:
        assert(False) # handle this case still

    plot_HTML += """
    scalex.domain([{}, {}]);
    scaley.domain([{}, {}]);
    """.format(x_range_min, x_range_max, y_range_min, y_range_max)

    # 2. Create the axes and set the axes names
    #=========================
    logger.debug('Plot generation: part 2')
    x_label = get_axis_label_name(inputs[0])
    if len(inputs) == 1:
        y_label = 'Response: {0}'.format(
                    persyst.system.primary_output_display_name_with_units)
    elif len(inputs) >= 2:
        y_label = get_axis_label_name(inputs[1])

    plot_HTML += """
    // Bottom x-axis
    svg.append("g")
        .attr("class", "x axis bottom")
        .attr("transform", "translate(0 ," + height + ")")
        .call(xAxis.orient("bottom"));

    // Bottom X-axis label
    svg.append("g")
        .attr("class", "x axis bottom label")
        .attr("transform", "translate(0," + (height + margin.bottom - deltabuffer) + ")")
        .append("text")
        .attr("font-family", "sans-serif")
        .attr("x", (width)/2)
        .attr("y", 0)
        .style("text-anchor", "middle")
        .text("{0}");

    // Top x-axis
    svg.append("g")
        .attr("class", "x axis top")
        .attr("transform", "translate(0, 0)")
        .call(xAxis.orient('top'));

    // Y-axis and y-axis label
    svg.append("g")
        .attr("class", "y axis left")
        .attr("transform", "translate(0, 0)")
        .call(yAxis.orient("left"));

    // Y-axis label
    svg.append("g")
        .attr("class", "y axis left label")
        .attr("transform", "translate(" + -margin.left/3*2 + "," + 0 + ")")
        .append("text")
        .attr("transform", "rotate(270)")
        .attr("class", "axislabel")
        .attr("font-family", "sans-serif")
        .attr("x", -height/2.0)
        .attr("y", -deltabuffer)
        .style("text-anchor", "middle")
        .text("{1}");

    // Y-axis right hand side
    svg.append("g")
        .attr("class", "y axis right")
        .attr("transform", "translate(" + width + "," + 0 + ")")
        .call(yAxis.orient("right"));
    """.format(x_label, y_label)


    # 3. Create the gridlines
    #=========================
    logger.debug('Plot generation: part 3')
    plot_HTML += """
    // X-axis gridlines
    svg.append("g")
        .attr("class", "x grid")
        .attr("transform", "translate(0, 0)")
        .call(xAxis
            .tickSize(-height, 0, 0)
            .tickFormat("")
        );

    // Y-axis gridlines
    svg.append("g")
        .attr("class", "y grid")
        .attr("transform", "translate(0, 0)")
        .call(yAxis
            .tickSize(width, 0, 0)
            .tickFormat("")
        );
    """

    # 4. Plot title
    #=========================
    logger.debug('Plot generation: part 4')
    plot_HTML += """
    // Chart title
    svg.append("g")
        .append("text")
        .attr("class", "rsm-plot title")
        // halfway between the plot and the outer edge
        .attr("transform","translate(" + (0) + "," + (-0.5*margin.top) + ")")
        .attr("x",(width/2.0))
        .attr("y",-deltabuffer)
        .attr("font-family","sans-serif")
        .attr("font-size","20px")
        .attr("fill","black")
        .attr("text-anchor","middle")
        .text("Summary of all experiments performed");
    """


    # 5. Now add the actual data points
    # TODO.v2: marker size proportional to response value
    #=========================

    logger.debug('Plot generation: part 5')
    plot_HTML += "\n    var rawdata = [\n"
    for idx, point in enumerate(x_data):
        plot_HTML += ('{{"x": {0}, "y": {1}, "rad": {2}, "col": "{3}", '
                      '"ord": "{4}", "resp": {5}}},\n')\
            .format(point, y_data[idx], 4, "black", idx+1, responses[idx])

    plot_HTML += "    ];\n"

    plot_HTML += """
    // Data is placed on top of the gridlines
    var circles = svg.append("g")
        .selectAll("circle")
        .data(rawdata)
        .enter()
        .append("circle")
        .attr("class","rsm-plot datapoints")
        .attr("cx",function(d){return scalex(d.x);})
        .attr("cy",function(d){return scaley(d.y);})
        .attr("r",function(d){return d.rad;})
        .attr("radius",function (d){ return d.rad;})
        .style("fill",function(d){return d.col;})
        .attr("ord",function(d){return d.ord;});
    """

    # Label the points in the plot
    # =======================
    # Use the defunct ``add_data_labels_NOT_USED`` function above,
    # or rather do this using Javascript

    # 6. Show the solution
    logger.debug('Plot generation: part 6')
    if show_solution:
    # =========================

        soldata = json.loads(persyst.solution_data)
        if len(inputs) == 1:
            # no need to check "continuous_dimensionality" anymore, once we
            # have defined the variable called "inputs"
            plot_HTML += "// Shows solution now \n\n"
            x_soldata = soldata['inputs'][inputs[0].slug.replace('-', '')]
            y_soldata = soldata['outputs']
            trio = [np.min(y_soldata), np.median(y_soldata), np.max(y_soldata)]
            plot_HTML += ("var colorScale = d3.scale.linear()"
                          ".range(['blue', 'green', 'red'])"
                          ".domain({0});".format(str(trio)))
            plot_HTML += "var soldata = [[\n"
            for idx, point in enumerate(x_soldata):
                plot_HTML += '{{"x":{0},"y":{1},"opcty":{2}}},\n'\
                    .format(point, y_soldata[idx], 1)

            plot_HTML += """    ]];

            // Coloured solution line idea from
            // http://bl.ocks.org/mbostock/1117287
                var linefunc = d3.svg.line()
                    .x(function(d) {
                        return scalex(d.x);
                    })
                    .y(function(d) {
                        return scaley(d.y);
                    })
                    .interpolate("linear");


                var solution = svg.append("g")
                    .attr("class", "rsm-plot solution");

                // Create a number of line segements from which to
                // construct the solution. One "g" per segment
                var solution_path = solution.selectAll("g")
                    .data(soldata)
                    .enter()
                    .append("g");

                function segments(values) {
                    var i = 0, n = values.length, segments = new Array(n - 1);
                    while (++i < n) {
                        segments[i - 1] = [values[i - 1], values[i]];
                    }
                    return segments;
                }

                var solution_pieces = solution_path.selectAll("path")
                    .data(segments)
                    .enter()
                    .append("path")
                    .attr("d", linefunc)
                    .attr("stroke-width", 4)
                    .attr("stroke-opacity", 0.5)
                    .style("stroke", function(d) {
                        return colorScale(d[0].y);
                    });
            """

        # 2D case here:
        if len(inputs)== 2:
            # no need to check "continuous_dimensionality" anymore, once we
            # have defined the variable called "inputs"

            X = soldata['inputs'][inputs[0].slug.replace('-', '')]
            Y = soldata['inputs'][inputs[1].slug.replace('-', '')]
            Z = soldata['outputs']

            # Rough rule of thumb for rounding
            x_round = int(np.ceil(4 - np.log10(np.max(X) - np.min(X))))
            y_round = int(np.ceil(4 - np.log10(np.max(Y) - np.min(Y))))

            # Using the matplotlib library here to generate contours.
            logger.debug('Plot generation: part 6a: loading library')
            fig = Figure()
            ax = fig.add_subplot(111)
            CS = ax.contour(X, Y, Z)

            logger.debug('Plot generation: part 6b: library used')
            levels = CS.levels.tolist()
            max_resp = np.max(Z)
            N = len(levels)
            logger.debug('Plot generation: part 6c: processing contours')

            # Add some extra levels based on a sqrt mapping (log(0.5) mapping)
            # i.e. find the values xx and yy below, given ``max_resp``
            # Value       Map   sqrt(Map)
            # levels[N-1] 2048  45
            # levels[N]   1024  32
            # xx          512   23
            # yy          256   16
            # max_resp    128   11
            slope1 = (1024 - 2048)/(levels[-1] - levels[-2] + 0.0)
            slope2 = (1024 - 128)/(levels[-1] - max_resp + 0.0)
            slope3 = (2048 - 128)/(levels[-2] - max_resp + 0.0)
            slope = (slope1 + slope2 + slope3)/3.0
            xx = levels[-2] -(2048 - 512)/slope
            yy = (xx + max_resp)/2.0
            levels.extend([xx, yy, (yy+max_resp+max_resp+max_resp)/4.0])
            off_peak = max_resp - 0.03*(max_resp - levels[-1])
            levels.extend([off_peak, max_resp, ])
            CS = ax.contour(X, Y, Z, levels=levels)
            logger.debug('Plot generation: part 6d: processing contours again.')
            colour = []

            # Now write the contour plot to D3 SVG code
            plot_HTML += "// Shows solution now \nvar soldata = \n[\n"
            for idx, cccontour in enumerate(CS.allsegs):
                colour.append(\
                    (np.round(CS.collections[idx].get_color()[0][0:3]*255))\
                    .tolist())
                plot_HTML += "\t[\n"
                for kontour in cccontour:
                    plot_HTML += "\t\t[\n"
                    for item in kontour:
                        ritem = item.round(3)
                        plot_HTML += '\t\t\t{{"x":{0},"y":{1}}},\n'\
                             .format(item[0].round(x_round),
                                     item[1].round(y_round))
                    plot_HTML += "\t\t],\n"
                plot_HTML += "\t],\n"
            plot_HTML += "];\n"

            if isinstance(CS.levels, list):
                plot_HTML += "var soln_levels = {0};\n".format(CS.levels)
            else:
                # Must be a numpy array
                plot_HTML += "var soln_levels = {0};\n".format(CS.levels.tolist())
            plot_HTML += "var soln_col = {0};\n".format(colour)

            plot_HTML += """
            var solution = svg.append("g")
                .attr("class", "rsm-plot solution");

            // Create a number of line segements from which to
            // construct the solution. One "g" per contour line
            var solution_path = solution.selectAll("g")
                .data(soldata)
                .enter()
                .append("g");

            var linefunc = d3.svg.line()
                .x(function(d) {
                    return scalex(d.x);
                })
                .y(function(d) {
                    return scaley(d.y);
                })
                .interpolate("cardinal");

                function segments(values) {
                    return values;
                }

            var solution_pieces = solution_path.selectAll("path")
                .data(segments)
                .enter()
                .append("path")
                .attr("d", linefunc)
                .attr("stroke-width", 2)
                .attr("stroke-opacity", 1)
                .attr("fill", "none")
                .style("stroke", function(d, subgroup, maingroup) {
                    var colour = soln_col[maingroup];
                    return d3.rgb(colour[0], colour[1], colour[2]);
                });
            """

            # TODO.v2 Put the gaps for the labels? Angle of the labels?
            # See ax.clabel(CS)




        plot_HTML += """
        // Data is placed on top of the gridlines
        var circles = svg.append("g")
            .selectAll("circle")
            .data(rawdata)
            .enter()
            .append("circle")
            .attr("class", "rsm-plot datapoints")
            .attr("cx", function (d) { return scalex(d.x); })
            .attr("cy", function (d) { return scaley(d.y); })
            .attr("r",  function (d) { return d.rad; })
            .attr("radius",  function (d) { return d.rad; })
            .style("fill", function(d) { return d.col; })
            .attr("ord", function (d) { return d.ord; });
        """

    # CS_lo = ax.contour(X1, X2, Y_lo, colors='#777777', levels=levels_lo,
    # baseline_xA, baseline_xB = transform_coords(x1=start_point[0],
    #                                             x2=start_point[1],
    #                                             rot=360-the_student.rotation)

    # 7. Add the legend
    # =============================
    logger.debug('Plot generation: part 7')
    plot_HTML += """
    if(showlegend){
	var legbox = svg.append("g")
            .attr("class", "legend")
            .attr("transform", "translate(" + (width+margin.right/4*1.5) + "," + height/2.0 + ")");

        var legend_square = 10;

        var legend = legbox.selectAll(".legend")
            .data(["Group 1", "Group 2"])
            .enter().append("g")
            .attr("class", "legenditem")
            .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

        // draw legend colored rectangles
        legend.append("rect")
            .attr("x", 0)
            .attr("width", legend_square)
            .attr("height", legend_square)
            .style("fill", "red");

        // draw legend text
        legend.append("text")
            .attr("x", legend_square+deltabuffer)
            .attr("y", legend_square/2.0)
            .attr("dy", ".35em")
            .style("text-anchor", "left")
            .text(function(d) { return d;})
    };
    """

    # 8. What to do when we mouse over the plot?
    # =============================
    logger.debug('Plot generation: part 8')
    plot_HTML += """
    // What to do when we mouse over a bubble
    var mouseOn = function() {
        var circle = d3.select(this);

        // Highlight the moused datapoint. The HTML #ID is defined in the table
        // template.
        $('#rsm-result-'+circle.attr('ord')).css( "background-color",
                                                   "rgb(128, 128, 255)" );
        circle.transition()
            .duration(800)
            .style("opacity", 1)
            .attr("r", parseInt(circle.attr('radius'))*2)
            .ease("elastic");

        // Append lines to bubbles that will be used to show the precise data
        // points. Translate their location based on margins.
        svg.append("g")
            .attr("class", "guide")
            .append("line")
            .attr("x1", circle.attr("cx"))
            .attr("x2", circle.attr("cx"))
            .attr("y1", +circle.attr("cy"))
            .attr("y2", height)
            .attr("stroke-width", 2)
            .style("stroke", "blue");  //circle.style("fill")

        svg.append("g")
            .attr("class", "guide")
            .append("line")
            .attr("x1", +circle.attr("cx"))
            .attr("x2", 0)
            .attr("y1", circle.attr("cy"))
            .attr("y2", circle.attr("cy"))
            .attr("stroke-width", 2)
            .style("stroke", "blue"); // circle.style("fill")

        // Function to move mouseover item to front of SVG stage, in case
        // another bubble overlaps it
        d3.selection.prototype.moveToFront = function() {
          return this.each(function() {
            this.parentNode.appendChild(this);
          });
        };
    };

    // What happens when we leave a bubble?
    var mouseOff = function() {
        var circle = d3.select(this);
        $('#rsm-result-'+circle.attr('ord')).css( "background-color", "" );

        // go back to original size and opacity
        circle.transition()
            .duration(800)
            .style("opacity", 1.0)
            .attr("r", parseInt(circle.attr("radius")))
            .ease("sin");

        // fade out guide lines, then remove them
        d3.selectAll(".guide")
            .transition()
            .duration(100)
            .styleTween("opacity", function() {
                return d3.interpolate(.5, 0); })
            .remove();
    };

    // The mousing functions
    circles.on("mouseover", mouseOn);
    circles.on("mouseout", mouseOff);
    """

    # 9. Close off and render the plot
    # =============================
    logger.debug('Plot generation: part 9')
    plot_HTML += """

    }  // End of the function: ``redraw_rsmchart``
    $(document).ready( function() {
        // Wait for DOM to be ready, otherwise you have DIV that has size of 0.
        redraw_rsmchart();
    });

    // Redraw char whenever the browser window is resized.
    window.addEventListener("resize", redraw_rsmchart);
    </script>
    """

    plot_out = ''
    for line in plot_HTML.split('\n'):
        if line.find('//') < 0:
            plot_out += line + '\n'
        else:
            plot_out += line[0:line.find('//')] + '\n'


    logger.debug('About to save the generated HTML.')
    persyst.plot_HTML = plot_out
    persyst.save()


def get_plot_and_data_HTML(persyst, input_set, show_solution=False):
    """Gets the data for plots, and then generates HTML code that may be
    rendered into the  Django template."""

    data, hash_value = get_person_experimental_data(persyst, input_set)
    expt_data = []
    expt = namedtuple('Expt', ['output', 'datetime', 'inputs'])
    for idx, output in enumerate(data['_output_']):

        # Simply skip points that cannot be revealed yet.
        if data['_datetime_'][idx] > datetime.now().replace(tzinfo=utc):
            continue

        input_item = {}
        for inputi in input_set:
            if inputi.ntype == 'CON':
                input_item[inputi.slug] = data[inputi.slug][idx]
            elif inputi.ntype == 'CAT':
                # Quick and dirty method to find the value that corresponds
                # to the one the user has:
                for key, value in json.loads(inputi.level_numeric_mapping).iteritems():
                    if value == data[inputi.slug][idx]:
                        input_item[inputi.slug] = key

        # Some experiments can't show their results just yet. Ensure that!

        item = expt(output=data['_output_'][idx],
                    datetime=data['_datetime_'][idx],
                    inputs=input_item)
        expt_data.append(item)

    # ``hash_value`` is None if .....
    if hash_value:
        if persyst.plot_HTML:
            # This speeds up page refreshes. We don't need to recreate existing
            # plots for a person/system combination.
            pass
        else:
            # The plot_HTML has been cleared; we're going to have to regenerate
            # the plot code.
            #logger.debug('Solution HTML about to be generated.')
            plot_wrapper(data, persyst, input_set, hash_value, show_solution)
            logger.debug('Solution HTML was generated.')
    else:
        assert(False)  # This shouldn't happen ever; since we add a baseline run
        # plot_html = 'No plot to display; please run an experiment first.'

    return expt_data


# UTILITY TYPE FUNCTIONS
def create_fake_usernames(number=10):
    """Chooses a humorous fake name (randomly created), for people to sign
    up with."""

    first_names = ["Sherpa", "Automatica", "Profit Hunter", "Optimum", "Guide",
                   "Optimizer", "Nimbostratus", "Stratus", "Cumulonimbus",
                   "Everest", "Kilimanjaro", "Amsterdam", "Amazon", "Nile",
                   "Yangtze", "Mississippi", "Pirana", "Shark", "Orca",
                   "Killer", "Lena", "Volga", "Danube", "Rio Grande", "Zambezi",
                   "Elephant", "Mouse", "Atomic", "Whale", "Rhino",
                   "Hippopotamus", "Girafe", "Mustang", "Kombi", "Crocodile",
                   "Turtle", "Ostrich", "Cassowary", "Nematode", "Isopod",
                   "Bug", "Roach", "Honey", "Hedgehog", "Sauropod", "Dystopian",
                   "Spider", "Lamprey", "Hagfish", "Sturgeon", "Trout",
                   "Bazooka", "AK47", "Canon", "Elvis", "Elton", "Sherlock",
                   "Inspector", "Detective", "Sergeant", "Hamlet", "Macbeth",
                   "Dexter", "Lancelot", "King", "Queen", "Prince", "Princess",
                   "Lord", "Lady", "Hercule", "Superintendent", "Napoleon",
                   "The", "Competitor", "Ninja", "Captain", "The one and only",
                   "Pinnacle", "Peak", "Ballpark", "Wild", "K2", "Lhotse",
                   "Kangchenjunga", "Manaslu", "Himalaya", "Atlas", "Thor",
                   "Sierra", "Boss", "CEO", "Principal", "Chief", "Kingpin",
                   "Honcho", "President", "Chair", "Director", "Chairman",
                   "Chairperson", "Leader", "Mistress", "Mister", "Monarch",
                   "Sovereign", "Head", "Trailblazer", "Trendsetter",
                   "The incredible", "Amazing", "Extraordinary", "Supreme",
                   "Bayesian", "Confounding", "Covariate",
                   ]

    last_names = ["Bayes", "Laplace", "Nightingale", "Galton", "Thiele",
                  "Pierce", "Pearson", "Gossett", "Fisher", "Bonferroni",
                  "Wilcoxon", "Neyman", "Deming", "Blackwell", "Tukey",
                  "Kendall", "Finetti", "Wold", "Hotelling", "Wishart",
                  "Anscombe", "Mosteller", "Federer", "Mahalanobis", "Markov",
                  "Snedecor", "Watson", "Jupiter", "Saturn",
                  "Weibull", "Marple", "Wimsey", "Dupin", "Holmes", "Marlowe",
                  "Poirot", "Magnum", "Millhone", "Dalgliesh", "Kojak", "Morse",
                  "Columbo", "Frost", "Clouseau", "CSI", "Quincy", "Nelson",
                  "Rebus", "Ruzzini", "Ducas", "Gaston", "Thundercat", "Mutant",
                  "Optimaxer", "Neptune", "Caucasus", "Alps",
                  "Ural", "Rockies", "Valhalla", "Denali", "Elbrus", "Rao",
                  "Spearman", "Taguchi", "Box", "Cox", "Wilcox", "Yates",
                  "Durbin", "Bose", "Norwood", "Shewhart", "Gauss", "Fuji",
                  "Bernoulli", "Friedman", "Hollerith", "Dantzig", "Rao",
                  "Kolmogorov", "Fermat", "Ontake", "Kita"]

    lone_names = ["Statstreet Boys", "I experiment thus I exist",
                  "Jessica Fletcher", "Tommy and Tuppence Beresford",
                  "John Thorndyke", "Hajime Kindaichi", "Amelia Peabody",
                  "Nancy Drew", "Miss Marple", "Veronica Mars",
                  "Joseph Rouletabille", "Louis XIV of France", "Shear Success",
                  "Tower of Babel", "Alt F4", "The Statistically Significants",
                  "The A-Team", "Knight Rider", "No Pie Charts Ever",
                  "Nanga Parbat", "Great Barrier Reef", "Take No Prisoners",
                  "Optimize Prime", "Dream Team", "Optimizer Prime",
                  "Numero Uno", "The Confouders", "The Standard Order",
                  "The Orderly Standard", "Standard Table", "The T-test",
                  "I am the Design", "I am Significant",
                  ]

    names = []
    for item in xrange(number+4):  # Generate a few extra
        first = first_names.pop(random.randint(0, len(first_names)-1))
        last = last_names.pop(random.randint(0, len(last_names)-1))
        names.append('{0} {1}'.format(first, last))

    names.append(lone_names.pop(random.randint(0, len(lone_names)-1)))

    pool = tuple(names)
    names = list(tuple(random.sample(pool, number+4)))

    # Now knock out the names that have been used already.
    for name in names:
        try:
            models.Person.objects.get(display_name=name)
            names.pop(names.index(name))
        except models.Person.DoesNotExist:
            pass

    return names[0:number]

def send_logged_email(subject, message, to_address_list):
    """ Sends an email to a user and it is assumed it is an HTML message.

    Returns a string error message if it failed. Returns None if sending
    succeeded.
    """
    from django.core.mail import send_mail
    logger.debug('Email [{0}]: {1}'.format(str(to_address_list), message))
    try:
        out = send_mail(subject=subject,
                  message=message,
                  from_email=None,
                  recipient_list=list(to_address_list),
                  fail_silently=False,
                  html_message=message)
        out = None
    except SMTPException as err:
        logger.error('EMAIL NOT SENT: {0}'.format(str(err)))
        out = str(err)
    return out

def inputs_to_JSON(inputs):
    """Converts the numeric inputs to JSON, after cleaning. This allows logging
    and storing them for all users.
    """
    copied = inputs.copy()
    copied.pop('email_address', None)
    #for key in copied.keys():
    #    if isinstance(key, basestring) and key.startswith('_'):
    #        copied.pop(key)
    return json.dumps(copied)

def generate_random_token(token_length, no_lowercase=True, check_unused=True):
    """Creates random length tokens from unmistakable characters."""
    choices = 'ABCEFGHJKLMNPQRSTUVWXYZ2345689'
    if not(no_lowercase):
        choices += 'abcdefghjkmnpqrstuvwxyz'


    while True:
        hashval = ''.join([random.choice(choices) for i in range(token_length)])
        if check_unused:
            try:
                models.Token.objects.get(hash_value=hashval)
            except models.Token.DoesNotExist:
                return hashval
            # It will repeat at this point

        else:
            return hashval

def csrf_failure(request, reason=""):
    """ Provide cleaner output when no cookies are present. """

    return HttpResponseForbidden(("Cookies are required for this website to "
        "function as expected. Please enable them, at least for this website."),
        content_type='text/html')

def serialize_numeric_dict(inputs):
    """ Serializes dictionary values, especially those that contain NumPy
    arrays.
    """
    for key, value in inputs.iteritems():
        if isinstance(value, np.ndarray):
            inputs[key] = value.tolist()

    return inputs
