# Dependancies (other than a basic Python + Django):
# pip install -U subprocess32
# pip install -U matplotlib
# pip install -U plotly

from django.shortcuts import get_object_or_404, render
from django.http import Http404, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.conf import settings as DJANGO_SETTINGS
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.db.utils import IntegrityError
from django.template.loader import render_to_string
from django.utils.timezone import utc

#import plotly
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
import matplotlib as matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Ideally put this in the Django settings file. Leave here for now.
WEBSITE_CORE = 'http://rsm.learnche.org'

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

class BadEmailInputError(RSMException):
    """ Raised when an email address is not valid."""
    pass

class BadEmailCannotSendError(RSMException):
    """ Raised when SMTPlib cannot send an email. """

class OutOfBoundsInputError(RSMException):
    """ Raised when an input is outside the bounds."""
    pass

def run_simulation(system, simvalues, rotation):
    """Runs simulation of the required ``System``, with timeout. Returns a
    result no matter what, even if it is the default result (failure state).
    """
    # If Python < 3.x, then we require the non-builtin library ``subprocess32``
    start_time = time.clock()

    code = "\nimport numpy as np\n" + system.source
    code_call = """\n\nout = simulate("""
    for key, value in simvalues.iteritems():
        code_call = code_call + "{0}={1}, ".format(key, value)

    code_call = code_call + ")\n"
    code = code + code_call

    if r"post_process(" in system.source:
        code = code + "print(post_process(out))"
    else:
        code = code + "print(out)"

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
    result = result.strip('\n')
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

    # TODO: adding the time-delay before results are displayed to the user.
    next_run.earliest_to_show = datetime.datetime.now()

    # TODO: add biasing for the user here
    return next_run

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
    #if request.session.test_cookie_worked():
        #request.session.delete_test_cookie()
        #return HttpResponse("You're logged in.")
    #else:
        #return HttpResponse("Please enable cookies and try again.")

    system = get_object_or_404(models.System, slug=short_name_slug)
    values = {}
    values_checked = {}
    try:
        # We do all our checks here, and if any fail an Exception is raised.
        # There are several checks: data entry, email validity, etc.

        # NB: Read the user values first before doing any checking on them.
        inputs = models.Input.objects.filter(system=system).order_by('slug')
        if request.session.get('signed_in', False):
            # Fix up here still; not sure what should happen in this branch.
            assert(False)
            values_checked['email_address'] = person.email
        else:
            values_checked['email_address'] = request.POST['email_address']

        for item in inputs:
            try:
                values[item.slug] = request.POST[item.slug]
            except KeyError:
                raise MissingInputError(('Input "{0}" was not specified.'.
                                         format(item.display_name)))

        # We've got all the inputs now; so validate them.
        values_checked.update(process_simulation_input(values, inputs))

        # Check the email address:
        try:
            validate_email(values_checked['email_address'])
        except ValidationError:
            raise(BadEmailInputError('You provided an invalid email address.'))

        # Success in checking the inputs. Create an input object for the user,
        # and run the experiment.
        # NOTE: ``next_run`` will not exist if an error was raised when
        #       generating that object.
        next_run = create_experiment_object(request, system, values_checked)


        # TODO: try-except path goes here to intercept time-limited experiments

    except (WrongInputError, OutOfBoundsInputError, MissingInputError,
            BadEmailInputError, BadEmailCannotSendError) as err:

        values['email_address'] = values_checked['email_address']
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


    # Clean-up the inputs by dropping any disallowed characters from the
    # function inputs:
    values_simulation = values_checked.copy()
    for key in values_simulation.keys():
        value = values_simulation.pop(key)
        key = key.replace('-', '')
        values_simulation[key] = value

    # Always pop out the email address; since
    values_simulation.pop('email_address', None)

    # TODO: Get the rotation here
    rotation = 0

    result, duration = run_simulation(system, values_simulation, rotation)
    next_run.time_to_solve = duration

    # Store the simulation results
    run_complete = process_simulation_output(result, next_run, system)
    run_complete.save()

    # Always return an HttpResponseRedirect after successfully dealing
    # with POST data. This prevents data from being posted twice if a
    # user hits the Back button.a
    return HttpResponseRedirect(reverse('rsmapp:show_one_system',
                                        args=(system.slug,)))

def show_one_system(request, short_name_slug, force_GET=False, extend_dict={}):
    """ Returns the details of one system that has been selected, including
    the leaderboard (generated by another function).

    ``force_GET`` and ``extend_dict`` are only used when the POST version,
    ``process_experiment(...)`` fails due to bad user input.
    """

    if request.POST and not(force_GET):
        return process_experiment(request, short_name_slug)

    # If it was not a POST request, but a (possibly forced) GET request...
    try:
        system = models.System.objects.get(slug=short_name_slug)
    except models.System.DoesNotExist:
        raise Http404(("Tried to find a system to optimize! But that system "
                       "does not exist."))

    fetch_leaderboard_results()


    # 1. User is signed in already.
    # 2. User is totally new; has filled in an expt, and is being returned
    #    here after filling their email address.
    # 3. As for #2, but it is an existing user.
    if request.session.get('signed_in', False):
        # Case 1.
        person = models.Person.objects.get(id=request.session['person_id'])
    elif request.session.get('send_new_user_email', False) or \
         request.session.get('send_returning_user_email', False):
        token_hash = request.session.get('token', None)
        if token_hash:
            token = models.Token.objects.get(hash_value=token_hash)
            person = token.person
        else:
            # Somehow we couldn't retrieve the token hash
            person = models.Person.objects.get(display_name='__Anonymous__',
                                                email='anonymous@learnche.org')
    else:
        # If all else fails (i.e. totally new visitor just visiting)
        person = models.Person.objects.get(display_name='__Anonymous__',
                                               email='anonymous@learnche.org')


    input_set = models.Input.objects.filter(system=system).order_by('slug')
    plot_data_HTML = get_plot_and_data_HTML(person, system, input_set)

    input_set, categoricals = process_simulation_inputs_templates(input_set,
                                                                  request,
                                                                  force_GET)
    context = {'system': system,
               'input_set': input_set,
               'person': person,
               'plot_html': plot_data_HTML[0],
               'data_html': plot_data_HTML[1]}
    context.update(extend_dict)   # used for the ``force_GET`` case when the
                                  # user has POSTed prior invalid data.
    context['categoricals'] = categoricals

    message = ''
    if request.session.get('send_new_user_email', False):
        message = ("Thanks for running your first experiment!<br>We have sent "
                   "you an email at {0} to get started. Please click on the "
                   "link in that email.<br><br>Only when you do that will the "
                   "results from your experiment be displayed.").format(\
                   person.email)
        # Do not allow the user to input any further new experiments
        context['input_set'] = []
    if request.session.get('send_returning_user_email', False):
        message =  ("Welcome back {0}. To sign yourself in, please click on "
                    "the link we sent to your email address: {1}. You can see "
                    "the result from your experiment once you do that.").format\
                    (person.display_name, person.email)
        # Do not allow the user to input any further new experiments
        context['input_set'] = []

    context['message'] = message
    return render(request, 'rsm/system-detail.html', context)

def validate_user(request, hashvalue):
    """ The new/returning user has been sent an email to sign in.
    Recall their token, mark them as validated, sign them in, run the experiment
    they had intended, and redirect them to the next URL associated with their
    token.

    If it is a new user, make them select a Leaderboard name first.
    """
    if request.POST:
        # Check the ``rsm_username`` is 4 or more characters, not taken already
        # not offensive
        # email me about the username being registered
        # complete the user with the new name,
        # make them a valid person .is_validated=True
        # save them,
        # return them to the system they came from
        # return HttpResponseRedirect(reverse('rsmapp:show_one_system',
    #                                    args=(token.system.slug,)))

        pass


    logger.info('Locating validation token {0}'.format(hashvalue))
    token = get_object_or_404(models.Token, hash_value=hashvalue)

    token.experiment.is_validated = True
    #token.experiment.save()

    token.was_used = True
    #token.save()

    #request.session['send_returning_user_email'] = True


    # or, maybe redirect to the "sign-in" function instead.
    #request.session['signed_in'] = True
    context = {'hashvalue': hashvalue,
               'message': 'Thank you for validating your email address.',
               'suggestions': create_fake_usernames(10),
               'person': token.person}


    return render(request, 'rsm/choose-new-leaderboard-name.html', context)

    #




    #suggest_names =



def send_suitable_email(person, send_new_user_email, send_returning_user_email,
                        hash_val):
    """ Either sends a validation email, or a log in email message. """
    if send_new_user_email:
        next_URI = '{0}/validate/{1}'.format(WEBSITE_CORE, hash_val)
        ctx_dict = {'validation_URI': next_URI}
        message = render_to_string('rsm/email_new_user_to_validate.txt',
                                   ctx_dict)

        subject = ("Confirm your email address for the Response Surface "
                   "Methods website!")
        to_address_list = [person.email.strip('\n'), ]

    if send_returning_user_email:
        next_URI = '{0}/sign-in/{1}'.format(WEBSITE_CORE, hash_val)
        ctx_dict = {'sign_in_URI': next_URI,
                    'username': person.display_name}
        message = render_to_string('rsm/email_sign_in_code.txt',
                           ctx_dict)
        subject = ("Unique email to sign-into the Response Surface "
               "Methods website.")
        to_address_list = [person.email.strip('\n'), ]

    # Use regular Python code to send the email in HTML format.
    message = message.replace('\n','\n<br>')
    return send_logged_email(subject, message, to_address_list), next_URI

def create_experiment_object(request, system, values_checked, person=None):
    """Create the input for the given user."""
     # TODO: check that the user is allowed to create a new input at this point
    #       in time. There might be a time limitation in place still.
    #       If so, offer to store the input and run it at the first possible
    #       occasion.

    # Once signed in: create 2 session settings: signed_in=True, person_id=``id``
    if request.session.get('signed_in', False):
        person = models.Person.objects.get(id=request.session['person_id'])
        validated_person = True
    else:
        validated_person = False
        # This is an anonymous (potentially new) user.
        # A: if the email exists, ask them to validate their experiment
        #    (create the experiment, but it is not validated yet)
        # B: if the email does not exist, again, create the experiment as
        #    unvalidated (is_validated=False), use ``anonymous[ID]`` as their
        #    leaderboard name, and send them a link to sign in with. That link
        #    will prompt them to create a username for the leaderboard, giving
        #    some interesting suggested names.

        send_new_user_email = False
        send_returning_user_email = False
        try:
            person = models.Person.objects.get_or_create(is_validated=False,
                                display_name='Anonymous',
                                email=values_checked['email_address'])
            person = person[0]
            person.display_name = person.display_name + str(person.id)
            person.save()
            send_new_user_email = True
        except IntegrityError as err:
            # The email address is not unique.
            person = models.Person.objects.get(
                                     email=values_checked['email_address'])
            send_returning_user_email = True

        # Store these two for use later on in the ``show_one_system()`` function
        request.session['send_new_user_email'] = send_new_user_email
        request.session['send_returning_user_email'] = send_returning_user_email


    # OK, we must have the person object now: whether signed in, brand new
    # user, or a returning user that has cleared cookies, or not been
    # present for a while.

    # If the user has not clicked on the email, place the experiment on
    # hold, until the user signs in.
    next_run = models.Experiment(person=person,
                                 system=system,
                                 inputs=inputs_to_JSON(values_checked),
                                 is_validated=validated_person,
                                 time_to_solve=-500,
                                 earliest_to_show=
    datetime.datetime(datetime.MAXYEAR, 12, 31, 23, 59, 59).replace(tzinfo=utc))
    next_run.save()

    hash_value = generate_random_token(10)
    token = models.Token(person=person,
                         system=system,
                         hash_value=hash_value,
                         experiment=next_run)
    request.session['token'] = token.hash_value

    failed, next_URI = send_suitable_email(person, send_new_user_email,
                                  send_returning_user_email, hash_value)
    if failed:
        next_run.delete()
        # token.delete()  :: not required. It hasn't been saved yet.
        # We do not want to use these 3 keys if the email address is wrong
        request.session.pop('send_new_user_email')
        request.session.pop('send_returning_user_email')
        request.session.pop('token')
        raise BadEmailCannotSendError("Couldn't send email: {0}".format(failed))
    else:
        token.next_URI = next_URI.strip(WEBSITE_CORE)
        token.save()
        return next_run

def fetch_leaderboard_results(system=None):
    """ Returns the leaderboard for the current system.
    """
    pass

def get_person_experimental_data(person, system, input_set):
    """Gets the data for a person and returns it, together with a hash value
    that should/is unique up to that point.

    The experiments are returned in the order they were run.
    """
    data = defaultdict(list)

    # Retrieve prior experiments which were valid, for this system, for person
    prior_expts = models.Experiment.objects.filter(system=system,
                                                person=person,
                                                is_validated=True,
                                                was_successful=True).order_by('earliest_to_show')
    data_string = str(person) + ';' + str(system)
    for entry in prior_expts:
        inputs = json.loads(entry.inputs)
        for item in input_set:
            data[item.slug].append(inputs[item.slug])
            data['_datetime_'].append(entry.earliest_to_show)

        # After processing all inputs, also process the response value:
        data['_output_'].append(entry.main_result)

    # Update the data_string
    data_string += str(data['_output_'])
    for item in input_set:
        data_string += str(data[item.slug])

    if not(data['_output_']):
        hash_value = token = None
    else:
        hash_value = hashlib.md5(data_string).hexdigest()

        # TODO: put this in a more suitable place: in the Experiment.hash_value
        #       field????
        token = models.Token.objects.get_or_create(person=person, system=system,
                                            hash_value=hash_value)
        token = token[0]
    return data, hash_value, token

def plot_wrapper(data, system, inputs, hash_value):
    """Creates a plot of the data, and returns the HTML code to display the
    plot"""

    USE_NATIVE = False
    USE_PLOTLY = not(USE_NATIVE)

    def plotting_defaults(vector, clamps=None):
        """ Finds suitable clamping ranges and a "dy" offset to place marker
        labels using heuristics.
        """
        if clamps is None:
            clamps = [float('-inf'), float('+inf')]
        y_min = max(min(vector), clamps[0])
        y_max = min(max(vector), clamps[1])
        y_range = y_max - y_min
        if y_range == 0.0:
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
        from matplotlib.figure import Figure  # for plotting
        matplotlib.rcParams['xtick.direction'] = 'out'
        matplotlib.rcParams['ytick.direction'] = 'out'
        fig = Figure(figsize=(9,7))
        rect = [0.15, 0.1, 0.80, 0.85] # Left, bottom, width, height
        ax = fig.add_axes(rect, frameon=True)
        marker_size = 20
    elif USE_PLOTLY:

        import plotly.plotly as py
        marker_size = 10
        fig, ax = plt.subplots()


    ax.set_title('Response surface: summary of all experiments performed',
                 fontsize=16)

    if len(inputs) == 1:
        x_data = data[inputs[0].slug]
        y_data = data['_output_']

        ax.set_xlabel(get_axis_label_name(inputs[0]),
                      fontsize=16)
        ax.set_ylabel(system.primary_output_display_name_with_units,
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
        from plotly.exceptions import PlotlyError
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
        from matplotlib.backends.backend_agg import FigureCanvasAgg
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


def get_plot_and_data_HTML(person, system, input_set):
    """Plots the data by generating HTML code that may be rendered into the
    Django template."""

    data, hash_value, token = get_person_experimental_data(person,
                                                           system,
                                                           input_set)

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

    if hash_value:
        if token and token.plot_HTML:
            plot_html = token.plot_HTML
        else:
            # This speeds up page refreshed. We don't need to recreate existing
            # plots for a person.
            plot_html = plot_wrapper(data, system, input_set, hash_value)
            token.plot_HTML = plot_html
            token.save()
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
                  "Snedecor", "Watson", "Jupiter", "Saturn"
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
            names.pop(name)
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
        out = str(err[0][to_address_list[0]])

    return out

def inputs_to_JSON(inputs):
    """Converts the numeric inputs to JSON, after cleaning. This allows logging
    and storing them for all users.
    """
    copied = inputs.copy()
    copied.pop('email_address', None)
    return json.dumps(copied)

def generate_random_token(token_length):
    """Creates random length tokens from unmistakable characters."""
    return ''.join([random.choice(('ABCEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvw'
                                   'xyz2345689')) for i in range(token_length)])