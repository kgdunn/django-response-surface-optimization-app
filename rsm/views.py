# Dependancies:
# pip install -U django
# pip install -U numpy
# pip install -U subprocess32
# pip install -U matplotlib
# pip install -U pillow
# pip install -U plotly

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

import sys
import time
import math
import json
import decimal
import random
import hashlib
import datetime
from smtplib import SMTPException
from collections import defaultdict, namedtuple
import logging
import numpy as np

# Ensure we can use Matplotlib in the background, on a headless machine
# This helps with Plotly
import matplotlib as matplotlib
if matplotlib.get_backend() != 'agg':
    matplotlib.use('agg')
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.exceptions import PlotlyError

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
    # If Python < 3.x, then we require the non-builtin library ``subprocess32``
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
            categoricals[item.slug] = json.loads(item.level_numeric_mapping)
            if force_GET:
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

def process_simulation_output(result, next_run, system):
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

    # TODO v2: adding the time-delay before results are displayed to the user.
    next_run.earliest_to_show = datetime.datetime.now().replace(tzinfo=utc)

    # TODO v2: add biasing for the user here
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
    system_list = models.System.objects.filter(is_active=True)
    person, enabled_status = get_person_info(request)

    context = {'system_list': system_list,
               'person': person,
               'enabled': enabled_status,
              }
    return render(request, 'rsm/root.html', context)

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


        # NOTE: ``next_run`` (below) will not exist if an error was raised in
        #       the above code when generating that object.

        # Success! at this point all inputs have been checked.
        # Create an input object for the user, and run the experiment.
        next_run = create_experiment_object(request, system, values_checked)

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

    next_run = execute_experiment_object(next_run, system, values_checked)

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
            logger.debug('REG-USER expt: {0}'.format(person.id))
        except models.Person.DoesNotExist:
            pass

    enabled_status = (person.is_validated == True) or \
                                        (person.display_name != '__Anonymous__')

    return person, enabled_status

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

    fetch_leaderboard_results()

    # Get the current ``person`` and ``enabled_status=True`` would indicate
    # it is an anonymous, or unvalidated person.
    person, enabled_status = get_person_info(request)

    # Get the relevant input objects for this system
    input_set = models.Input.objects.filter(system=system).order_by('slug')

    if enabled_status:
        # Have there been any prior experiments for this person?
        if models.Experiment.objects.filter(system=system, person=person,
                                            was_successful=True).count() == 0:

            # If not: Create a baseline run for the person at the default values
            default_values = {}
            for inputi in input_set:
                default_values[inputi.slug] = inputi.default_value

            baseline = create_experiment_object(request, system,
                                                default_values, person)
            baseline = execute_experiment_object(baseline, system,
                                                 default_values)

        # Initiate the ``PersonSystem`` for this combination only once
        persysts = models.PersonSystem.objects.filter(system=system,
                                                                 person=person)

        # If this is zero, it is because it is the first time the person has
        # visited this system
        if persysts.count() == 0:
            future = datetime.datetime(datetime.MAXYEAR, 12, 31, 23, 59, 59).\
                                                           replace(tzinfo=utc)
            solution_date = datetime.datetime.now() + \
                       datetime.timedelta(0, system.max_seconds_before_solution)
            solution_date = solution_date.replace(tzinfo=utc)

            persyst = models.PersonSystem(person=person, system=system,
                                          completed_date=future, frozen=False,
                                          show_solution_as_of=solution_date)
            persyst.save()
        else:
            persyst = persysts[0]

        # Should we show the solution? Let's check:
        if datetime.datetime.now().replace(tzinfo=utc) > \
                               persyst.show_solution_as_of.replace(tzinfo=utc):
            show_solution = True

            # Have we generated the solution before? If so, don't do it again.
            # But if not, generate it once, and then save it (it is expensive).
            if persyst.frozen == False:
                persyst = generate_solution(persyst)

            # Freeze it once the solution has been generated.
            persyst.frozen = True
            persyst.save()
        else:
            show_solution = False

        # Only get this for signed in users
        plot_data_HTML = get_plot_and_data_HTML(persyst, input_set,
                                                show_solution)

    else:
        plot_data_HTML = [[ ], [ ]]

    # if-else-end: if enabled_status

    input_set, categoricals = process_simulation_inputs_templates(input_set,
                                                                  request,
                                                                  force_GET)
    context = {'system': system,
               'input_set': input_set,
               'person': person,
               'enabled': enabled_status,
               'plot_html': plot_data_HTML[0],
               'data_html': plot_data_HTML[1]}
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
        message = ('That validation key has been used already. Please request '
                   'another by clicking on the "Sign-in" button')
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

def create_experiment_object(request, system, values_checked, person=None):
    """Create the input for the given user."""
    # TODO: check that the user is allowed to create a new input at this point
    #       in time. There might be a time limitation in place still.
    #       If so, offer to store the input and run it at the first possible
    #       occasion.

    if request.session.get('person_id', False):
        person = models.Person.objects.get(id=request.session['person_id'])
    else:
        logger.error('Unlogged user attempted to create an experiment.')
        assert(False)

    next_run = models.Experiment(person=person,
                                 system=system,
                                 inputs=inputs_to_JSON(values_checked),
                                 time_to_solve=-500,
                                 earliest_to_show=
    datetime.datetime(datetime.MAXYEAR, 12, 31, 23, 59, 59).replace(tzinfo=utc))
    return next_run

def execute_experiment_object(expt_obj, system, values_checked):
    """Typically called after ``create_experiment_object`` once all inputs
    have been cleaned and checked."""
    # Clean-up the inputs by dropping any disallowed characters from the
    # function inputs:
    values_simulation = values_checked.copy()
    for key in values_simulation.keys():
        value = values_simulation.pop(key)
        key = key.replace('-', '')
        values_simulation[key] = value

    # TODO.v2: Get the rotation here
    rotation = 0

    result, duration = run_simulation(system, values_simulation)
    expt_obj.time_to_solve = duration

    # Finally, store the simulation results and return the experimental object
    expt_obj = process_simulation_output(result, expt_obj, system)
    expt_obj.save()
    return expt_obj

def generate_solution(persyst):
    """Generates a solution for a given Person/System combination. This is
    expensive process, so save the results. """
    system = persyst.system
    input_set = models.Input.objects.filter(system=system).order_by('slug')

    solution_inputs = {}
    RESOLUTION = 50
    for inputi in input_set:
        input_name = inputi.slug.replace('-', '')
        solution_inputs[input_name] = np.linspace(start=inputi.plot_lower_bound,
                                                  stop=inputi.plot_upper_bound,
                                                  num=RESOLUTION,
                                                  endpoint=True)

    # TODO.v2: Apply the necessary rotation here
    rotation = persyst.rotation

    # Ignore post_process(): which is where the noise is added.
    result, duration = run_simulation(system, solution_inputs,
                                      show_solution=True)

    # Finally, store the simulation results and return the experimental object
    class FakeClass(): pass
    results = FakeClass()
    results = process_simulation_output(result, results, system)
    solution_data = {}
    solution_data['inputs'] = serialize_numeric_dict(solution_inputs)
    if results.was_successful:
        solution_data['outputs'] = results.main_result
    else:
        logger.error('Error generating solution for {0}'.format(persyst))
        assert(False)

    persyst.solution_data = json.dumps(solution_data, allow_nan=True)
    return persyst

def fetch_leaderboard_results(system=None):
    """ Returns the leaderboard for the current system.
    """
    pass

def get_person_experimental_data(persyst, input_set):
    """Gets the data for a person and returns it, together with a hash value
    that should/is unique up to that point.

    The experiments are returned in the order they were run.
    """
    data = defaultdict(list)

    # Retrieve prior experiments which were successful, for this system,
    # for the current logged in person.
    prior_expts = models.Experiment.objects.filter(system=persyst.system,
                                                   person=persyst.person,
                            was_successful=True).order_by('earliest_to_show')
    data_string = str(persyst.person) + ';' + str(persyst.system)
    for entry in prior_expts:
        inputs = json.loads(entry.inputs)
        for item in input_set:
            data[item.slug].append(inputs[item.slug])
            data['_datetime_'].append(entry.earliest_to_show)

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

def plot_wrapper(data, system, inputs, hash_value, show_solution=False):
    """Creates a plot of the data, and returns the HTML code to display the
    plot.
    Optionally shows the solution if ``show_solution`` is True.
    """
    USE_NATIVE = False
    USE_PLOTLY = not(USE_NATIVE)
    USE_JSON = True

    def plotting_defaults(vector, clamps=None):
        """ Finds suitable clamping ranges and a "dy" offset to place marker
        labels using heuristics.
        """
        finite_clamps = True
        if clamps is None:
            clamps = [float('-inf'), float('+inf')]
            finite_clamps = False

        y_min = max(min(vector), clamps[0])
        y_max = min(max(vector), clamps[1])
        y_range = y_max - y_min
        if y_range == 0.0 and finite_clamps:
            y_min, y_max = clamps
            y_range = y_max - y_min
        elif y_range == 0.0 and not(finite_clamps):
            y_range = 1.0


        y_range_min, y_range_max = y_min - 0.07*y_range, y_max + 0.07*y_range
        dy = 0.015*y_range
        return (y_range_min, y_range_max, dy)

    def get_axis_label_name(input_item):
        """Returns an axis label for a particular input."""
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

    def add_labels(axis, dims, xvalues, yvalues, zvalues=None, dx=0, dy=0, dz=0,
                   rotate=False):
        """Adds labels to an axis ``axis`` in 1, 2 or 3 dimensions.
        If ``rotate`` is True, then the labels are randomly rotated about the
        plotted point. The ``dx``, ``dy`` and ``dz`` are the base offsets from
        the coordinate values given in ``xvalues``, ``yvalues``, and ``zvalues``
        respectively.
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

    # 1. Get the limits of the plot from the inputs
    # 2. Create the title automatically
    # 3. Get the axis names from the inputs
    # 4a. Plot the scatterplot of the data
    # 4b: use a marker size proportional to objective function
    # 5. Add labels to each point
    # 6. Add gridlines

    # Create the figure
    if USE_NATIVE:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        matplotlib.rcParams['xtick.direction'] = 'out'
        matplotlib.rcParams['ytick.direction'] = 'out'
        fig = Figure(figsize=(9,7))
        rect = [0.15, 0.1, 0.80, 0.85] # Left, bottom, width, height
        ax = fig.add_axes(rect, frameon=True)
        marker_size = 20

    elif USE_PLOTLY:

        marker_size = 10
        fig, ax = plt.subplots()

    ax.set_title('Response surface: summary of all experiments performed',
                 fontsize=16)

    if len(inputs) == 1:
        x_data = data[inputs[0].slug]
        y_data = data['_output_']

        ax.set_xlabel(get_axis_label_name(inputs[0]),
                      fontsize=16)
        ax.set_ylabel('Response: {0}'.format(\
                            system.primary_output_display_name_with_units),
                      fontsize=16)

        x_range_min, x_range_max, dx = plotting_defaults(x_data,
            clamps=[inputs[0].plot_lower_bound, inputs[0].plot_upper_bound])

        y_range_min, y_range_max, dy = plotting_defaults(data['_output_'])

        ax.set_xlim([x_range_min, x_range_max])
        ax.set_ylim([y_range_min, y_range_max])


    elif len(inputs) == 2:
        # To ensure we always present the data in the same way
        inputs = inputs.order_by('slug')

        x_data = data[inputs[0].slug]
        y_data = data[inputs[1].slug]

        ax.set_xlabel(get_axis_label_name(inputs[0]), fontsize=16)
        ax.set_ylabel(get_axis_label_name(inputs[1]), fontsize=16)

        x_range_min, x_range_max, dx = plotting_defaults(x_data,
                clamps=[inputs[0].plot_lower_bound, inputs[0].plot_upper_bound])
        y_range_min, y_range_max, dy = plotting_defaults(y_data,
                clamps=[inputs[1].plot_lower_bound, inputs[1].plot_upper_bound])

        ax.set_xlim([x_range_min, x_range_max])
        ax.set_ylim([y_range_min, y_range_max])

    elif len(inputs) >= 3:
        pass

    # Now add the actual data points
    if len(inputs) == 1:
        ax.plot(x_data, y_data, 'k.', ms=marker_size)

    elif len(inputs) == 2:

        # TODO: marker size proportional to response value
        ax.plot(x_data, y_data, 'k.', ms=marker_size)

    # Label the points in the plot
    add_labels(ax, len(inputs), x_data, y_data, dx=dx, dy=dy, rotate=False)


    #if show_result:
        #r = 70         # resolution of surface
        #x1 = np.arange(limits_A[0], limits_A[1], step=(limits_A[1] - limits_A[0])/(r+0.0))
        #x2 = np.arange(limits_B[0], limits_B[1], step=(limits_B[1] - limits_B[0])/(r+0.0))
        #X3_lo = 'H'
        #X3_hi = 'X'

        #X1, X2 = np.meshgrid(x1, x2)
        #Y_lo, Y_lo_noisy = generate_result(the_student, (X1, X2, X3_lo),
                                           #pure_response=True)
        #Y_hi, Y_hi_noisy = generate_result(the_student, (X1, X2, X3_hi),
                                           #pure_response=True)

        #levels_lo = np.linspace(-30, 3000, 55)*1
        #levels_hi = np.linspace(-30, 3101, 55)*1
##
## DO NOT SHOW BLACK CONTOURS
##
##        CS_lo = ax.contour(X1, X2, Y_lo, colors='#777777', levels=levels_lo,
##                           linestyles='solid', linewidths=1)
        #CS_hi = ax.contour(X1, X2, Y_hi, colors='#FF0000', levels=levels_hi,
                           #linestyles='dotted', linewidths=1)
##        ax.clabel(CS_lo, inline=1, fontsize=10, fmt='%1.0f' )
        #ax.clabel(CS_hi, inline=1, fontsize=10, fmt='%1.0f' )


    # Plot constraint
    #ax.plot([constraint_a[0], constraint_b[0]], [constraint_a[1], constraint_b[1]], color="#EA8700", linewidth=2)

    #baseline_xA, baseline_xB = transform_coords(x1=start_point[0], x2=start_point[1],
    #                                            rot=360-the_student.rotation)
    #my_logger.debug('Baseline [%s] = (%s, %s)' % (the_student.student_number,
    #                                              baseline_xA, baseline_xB))

    #ax.text(391, 20.5, the_student.group_name, horizontalalignment='left', verticalalignment='center', fontsize=10, fontstyle='italic')

    # Baseline marker and label.
    #ax.text(baseline_xA, baseline_xB, "    Baseline",
    #        horizontalalignment='left', verticalalignment='center',
    #        color="#0000FF")
    #ax.plot(baseline_xA, baseline_xB, 'r.', linewidth=2, ms=20)

    #for idx, entry_A in enumerate(factor_A):

        ## Do not rotate the location of the labels
        #xA, xB = entry_A, factor_B[idx]
        #if factor_C[idx] == 'H':
            #ax.plot(xA, xB, 'k.', ms=20)
        #else:
            #ax.plot(xA, xB, 'r.', ms=20)


        #rand_theta = random.uniform(0, 2*np.pi)
        #dx = 1.4 * np.cos(rand_theta) # math.copysign(random.uniform(0.45, 0.55), random.uniform(-1,1))
        #dy = 1.0 * np.sin(rand_theta) # math.copysign(random.uniform(0.45, 0.55), random.uniform(-1,1))

        #ax.text(xA+dx, xB+dy, str(idx+1),horizontalalignment='center', verticalalignment='center',)

    # 6. Grid lines
    ax.grid(color='k', linestyle=':', linewidth=1)

    if USE_PLOTLY:
        logger.debug('Begin: generating Plotly figure: ' + hash_value)
        try:
            plot_url = py.plot_mpl(fig,
                                   filename=hash_value,
                                   fileopt='overwrite',
                                   auto_open=False,
                                   sharing='public')
        except PlotlyError as e:
            logger.error('Failed to generate Plotly plot:{0}'.format(e.message))
            return ('A plotting error has occurred and has been logged. However'
                    ', for faster response, please inform kgdunn@gmail.com. '
                    'Thank you.')

        plot_HTML = """<iframe frameborder="0" seamless="seamless"
            autosize="true" width=100% height=600 modebar="false"
            src="{0}.embed"></iframe>""".format(plot_url)

        logger.debug('Done : generating Plotly figure: ' + plot_url)

    elif USE_NATIVE:

        canvas=FigureCanvasAgg(fig)
        logger.debug('Saving figure: ' + hash_value)
        fig.savefig(hash_value+'.png',
                    dpi=150,
                    facecolor='w',
                    edgecolor='w',
                    orientation='portrait',
                    papertype=None,
                    format=None,
                    transparent=True)
        plot_HTML = hash_value+'.png'

    return plot_HTML


def get_plot_and_data_HTML(persyst, input_set, show_solution=False):
    """Plots the data by generating HTML code that may be rendered into the
    Django template."""

    data, hash_value = get_person_experimental_data(persyst, input_set)
    expt_data = []
    expt = namedtuple('Expt', ['output', 'datetime', 'inputs'])
    for idx, output in enumerate(data['_output_']):
        input_item = {}
        for inputi in input_set:
            input_item[inputi.slug] = data[inputi.slug][idx]

        item = expt(output=output,
                    datetime= data['_datetime_'][idx],
                    inputs=input_item)
        expt_data.append(item)

    # ``hash_value`` is None if .....
    if hash_value:
        if persyst.plot_HTML:
            # This speeds up page refreshes. We don't need to recreate existing
            # plots for a person/system combination.
            plot_html = plothash.plot_HTML
        else:
            # The plot_HTML has been cleared; we're going to have to regenerate
            # the plot code.
            persyst.plot_HTML = plot_wrapper(data, persyst.system, input_set,
                                             hash_value, show_solution)
            plothash.save()
            plot_html = persyst.plot_HTML
    else:
        plot_html = 'No plot to display; please run an experiment first.'

    return plot_html, expt_data


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
