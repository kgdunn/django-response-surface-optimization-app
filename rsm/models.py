from django.db import models
import numpy as np
import datetime
import json
from django.utils.timezone import utc
from django.utils.text import slugify

class Person(models.Model):
    """ Defines a person / course participant """
    display_name = models.CharField(max_length=200,
                                    verbose_name="Leaderboard name")
    slug = models.SlugField(default='')
    level = models.SmallIntegerField(verbose_name="Skill level of the user",
                                     blank=False, null=False, default=1)
    email = models.EmailField(unique=True)
    is_validated = models.BooleanField(default=False, help_text=('Will be auto-'
                        'validated once user has clicked on their email link.'))

    def __str__(self):
        return '{0} [{1}]; level={2}'.format(self.display_name, self.email,
                                             self.level)

    def save(self, *args, **kwargs):
        # Force the bounds to be compatible
        self.display_name  = self.display_name.strip()
        self.slug = slugify(self.display_name)
        super(Person, self).save(*args, **kwargs) # Call the "real" save()

class PersonSystem(models.Model):
    """ Changes made to a System for a specific Person."""
    person = models.ForeignKey('Person')
    system = models.ForeignKey('System')
    rotation = models.TextField(default='', blank=True,
                            help_text='Rotation around axis for this system')
    offset_y = models.FloatField(default=0.0, blank=True,
                                verbose_name="Offset for system output")

    # When did the user initiate completion of the system?
    completed_date = models.DateTimeField()
    # At this point onwards the user is considered to have solved it.
    show_solution_as_of = models.DateTimeField()
    frozen = models.BooleanField(default=False, help_text=('If true, prevents '
                        'any further additions to this system.'))
    started_on = models.DateTimeField(auto_now_add=True)
    solution_data = models.TextField(help_text='In JSON format', blank=True,
                                     default='')
    # Plots are expensive to draw; hash them this way to cache them.
    plot_hash = models.CharField(max_length=32, editable=False, default='-'*32)
    plot_HTML = models.TextField(default='', blank=True)

    # JSON field that shows the leaderboard breakdown and scores:
    # It is a list of score dictionaries, to track the leaderboard trajectory
    # of the user over time.
    # [[{"score": -1.0}, datetime], [{...}, datetime], etc]
    #
    # The "datetime" refers to when the experiment may be revealed to the
    leaderboard = models.TextField(default='[[{"score": -1.0}, 0]]', blank=True)

    # User notes are a place for the user to share information about what they
    # have done.
    user_notes = models.TextField(blank=True, default='')


    def __str__(self):
        return '{0} [{1}]'.format(self.system.full_name,
                                  self.person.display_name, )

    def has_solved(self):
        """Determines if the system has been solved."""
        return self.completed_date < datetime.datetime.now().replace(tzinfo=utc)
    is_solved = property(has_solved)

    def get_score(self):
        try:
            history = json.loads(self.leaderboard)
            count = -1
            while (-count) < len(history):
                date = datetime.datetime.strptime(history[count][1][0:19],
                                                         "%Y-%m-%dT%H:%M:%S")
                if date <= datetime.datetime.now():
                    return history[count][0].get('score', -1.0)

                # We should break out the loop after at least 2 iterations
                count -= 1

            # Safety net
            return -1.0

        except (KeyError, ValueError):
            return -1.0

class Token(models.Model):
    """ Tokens capture time/date and permissions of a user to access the
    ``System`` models.
    """
    person = models.ForeignKey('rsm.Person', null=True, blank=True)
    system = models.ForeignKey('rsm.System', null=True, blank=True)
    hash_value = models.CharField(max_length=32, editable=False, default='-'*32)
    was_used = models.BooleanField(default=False)
    time_used = models.DateTimeField(auto_now=True, auto_now_add=False)
    # Use tokens to redirect a ``Person`` to a next web page.
    next_URI = models.CharField(max_length=50, editable=True, default='',
                                blank=True)
    experiment = models.ForeignKey('rsm.Experiment', blank=True, null=True)

class Tag(models.Model):
    """ Tags for ``Systems``. """
    short_name = models.CharField(max_length=50)
    description = models.CharField(max_length=150)

    def __str__(self):
        return self.short_name

class Experiment(models.Model):
    """ The inputs, and the corresponding result(s) from simulating the system
    for a particular user. """
    person = models.ForeignKey('rsm.Person')
    system = models.ForeignKey('rsm.System')

    time_to_solve = models.FloatField(verbose_name="Time to solve model",
                                      blank=False, null=False, default=0.0)
    earliest_to_show = models.DateTimeField(
        verbose_name="Don't show the result before this point in time")

    # True if the result is successfully simulated (i.e. if the simulation
    # did not time out, or crash for some reason.)
    was_successful = models.BooleanField(default=False)
    inputs = models.TextField(verbose_name=("The system inputs logged in JSON "
                                            "format"))
    main_result = models.FloatField(verbose_name="Primary numeric output",
                                   default=-987654321.0)
    other_outputs = models.TextField(verbose_name=("Other outputs produced, "
                                                   "including string messages, "
                                                   "in JSON format, as defined "
                                                   "by the ``System``."),
                                                   blank=True, null=True)
    hash_value = models.CharField(max_length=32, editable=False, default='-'*32)

    def __str__(self):
            return 'System: {0}: {1}'.format(self.system.slug, str(self.inputs))

class System(models.Model):
    """ A simulated system, or process. """
    full_name = models.CharField(max_length=250)
    slug = models.SlugField()
    description = models.TextField(verbose_name=("A description of what this "
        "system does, and hints on the objective of the optimization."),
        unique=True, blank=False, null=False, default="The aim of this ...")
    is_active = models.BooleanField(default=False, help_text=("If False, then "
            "this system will not be usable."))

    image_description = models.ImageField(null=True, upload_to='rsm')
    image_source_URL = models.CharField(max_length=500, default='')
    level = models.FloatField(verbose_name=("Skill level required by user"),
                                     blank=False, null=False, default=0)
    source = models.TextField(verbose_name=("Python source code that will be "
                                           "executed. A function with the "
                                           "name ``simulate(...)`` must exist. "
                                           "The NumPy library is available "
                                           "as ``np``."),
                              default=u"def simulate(**inputs):\n  # Code here",
                              unique=True, blank=False)
    simulation_timeout = models.PositiveSmallIntegerField(blank=False,
                                                          null=False,
                                                          default=5,
        verbose_name="Seconds that may elapse before simulation is killed.")
    default_error_output = models.FloatField(default=-987654321.0,
        verbose_name="The default value assigned when the simulation fails.")
    n_inputs = models.PositiveSmallIntegerField(verbose_name=("Number of model "
                                                "inputs"), default=1)

    n_outputs = models.PositiveSmallIntegerField(verbose_name=("Number of model "
                                                "outputs"), default=1)
    primary_output_display_name_with_units = models.TextField(
        default="Response value")
    output_json = models.TextField(verbose_name=("Comma-separated list of model "
        'output names; the first one must be "result"'), default="result")

    # Some systems will have time delays before the results are available.
    delay_result = models.IntegerField(verbose_name=("Number of seconds before "
                "the result of a SINGLE experiment may be shown to users."), default=0)
    message_while_waiting = models.CharField(max_length=510, default='',
                                             help_text="Message to display while 'running' the experiment.'")

    tags = models.ManyToManyField('rsm.Tag')

    system_notes = models.TextField(blank=True)
    known_optimum_response = models.FloatField(default=-9999999999)
    cost_per_experiment = models.FloatField(help_text="Dollar cost per run",
                                            default=10.00)
    min_experiments_allowed = models.PositiveIntegerField(default=5)
    max_experiments_allowed = models.PositiveIntegerField(default=100)
    max_seconds_before_solution = models.PositiveIntegerField(default=2147483647,
        help_text=('Max seconds to wait before showing the solution. 43200=30 '
                   'days, as an example.'))

    offset_y_range = models.CharField(max_length=100,
                                      help_text='JSON: [Two values, in a list]')

    def continuous_dimensionality(self):
        """ Determines how many continuous variables are in the system.
            Used to calculate rotations.
        """
        input_set = Input.objects.filter(system=self)
        dimensionality = 0
        for inputi in input_set:
            if inputi.ntype == 'CON':
                dimensionality += 1

        return dimensionality

    def __str__(self):
        return self.full_name

    def save(self, *args, **kwargs):
        # Force the min and max experiment numbers to be consistent
        assert(self.min_experiments_allowed < self.max_experiments_allowed)
        super(System, self).save(*args, **kwargs) # Call the "real" save()



class Input(models.Model):
    """An input into one of the systems"""
    display_name = models.CharField(max_length=200)
    slug = models.SlugField()
    system = models.ForeignKey('rsm.System')
    NUMERIC_TYPE_CHOICES = (
        ('CON', 'Continuous'),
        ('CAT', 'Categorical'),
    )
    ntype = models.CharField(choices=NUMERIC_TYPE_CHOICES, max_length=3,
                             default='CON', verbose_name=("The numeric type of "
                                                        "the input variable."))
    level_numeric_mapping = models.TextField(verbose_name=("Specify UNIQUE "
        "names for each numeric level of a categorical variable; JSON format."),
        blank=True, help_text=('For example: {"water": -1, "vinegar": +1} '
                               'would map the "water" level to -1 and "vinegar" '
                               'level to +1 in the simulation. Leave blank for '
                               'continuous variables.'))
    lower_bound = models.FloatField(blank=True, help_text=("If supplied, will "
        "ensure the user does not enter a value below this."), null=True)
    upper_bound = models.FloatField(blank=True, help_text=("If supplied, will "
            "ensure the user does not enter a value above this."),
                                    null=True)
    plot_lower_bound = models.FloatField(blank=False, default=0.0,
        help_text=("Plots must be generated to show the true solution. What is "
                   "the lower used in these plots for this variable? (Leave "
                   "as zero for categorical variables.)"))
    plot_upper_bound = models.FloatField(blank=False, default=0.0,
        help_text=("Plots must be generated to show the true solution. What is "
                   "the upper used in these plots for this variable? (Leave "
                   "as zero for categorical variables.)"))

    default_value = models.FloatField(help_text=("The default used, e.g. in a "
        "multidimensional (>3) plot. For categorical variables this MUST "
        "correspond to one of the levels in the JSON dictionary."))
    units_prefix = models.CharField(max_length=100, help_text=("The prefix for "
            "the units of the input (can be blank)"), blank=True, null=True)
    units_suffix = models.CharField(max_length=100, help_text=("The suffix for "
            "the units of the input (can be blank)"), blank=True, null=True)
    error_message = models.CharField(max_length=200, blank=True,
                                     help_text=("Any error message text that "
                                                "should be shown during input "
                                                "validation."))
    n_decimals = models.PositiveSmallIntegerField(help_text=("The number of "
        "decimals to show in the numeric representation. Not applicable for "
        "categorical variables (leave as 0.0)"),
        verbose_name="Number of decimals", default=0)

    def save(self, *args, **kwargs):
        # Force the bounds to be compatible
        if self.lower_bound:
            self.plot_lower_bound = min(self.plot_lower_bound, self.lower_bound)
        if self.upper_bound:
            self.plot_upper_bound = max(self.plot_upper_bound, self.upper_bound)
        if self.lower_bound and self.upper_bound:
            if self.lower_bound > self.upper_bound:
                assert(False)
        if self.plot_lower_bound > self.plot_upper_bound:
            assert(False)


        super(Input, self).save(*args, **kwargs) # Call the "real" save() method

    def __str__(self):
        return self.system.full_name + "::" + self.display_name

