from django.db import models
import numpy as np


class Person(models.Model):
    """ Defines a person / course participant """
    display_name = models.CharField(max_length=200,
                                    verbose_name="Leaderboard name")
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
        super(Person, self).save(*args, **kwargs) # Call the "real" save()


#class PersonSystem(models.Model):
#    """ Changes made to a System for a specific Person."""
#    rotation

class Token(models.Model):
    """ Tokens capture time/date and permissions of a user to access the
    ``System`` models.
    """
    person = models.ForeignKey('rsm.Person')
    system = models.ForeignKey('rsm.System', null=True, blank=True)
    hash_value = models.CharField(max_length=32, editable=False, default='-'*32)
    was_used = models.BooleanField(default=False)
    time_used = models.DateTimeField(auto_now=True, auto_now_add=False)
    # Use tokens to redirect a ``Person`` to a next web page.
    next_URI = models.CharField(max_length=50, editable=True, default='',
                                blank=True)
    experiment = models.ForeignKey('rsm.Experiment', blank=True, null=True)

class PlotHash(models.Model):
    """ Plots are expensive to draw; hash them this way to cache them.
    """
    person = models.ForeignKey('rsm.Person')
    system = models.ForeignKey('rsm.System')
    hash_value = models.CharField(max_length=32, editable=False, default='-'*32)
    time_last_used = models.DateTimeField(auto_now=True, auto_now_add=False)
    plot_HTML = models.TextField(default='', blank=True)

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
    # True if the result is successfully simulated (i.e. if the simulation
    # did not time out, or crash for some reason.)
    is_validated = models.BooleanField(help_text=("False: indicates the Person "
                    "has not validated their choice by signing in (again)."),
                    default=False)
    delete_by = models.DateTimeField(auto_now=True,
        verbose_name="Delete the experiment at this time if not validated.")
    time_to_solve = models.FloatField(verbose_name="Time to solve model",
                                      blank=False, null=False, default=0.0)
    earliest_to_show = models.DateTimeField(
        verbose_name="Don't show the result before this point in time")
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

class System(models.Model):
    """ A simulated system, or process. """
    full_name = models.CharField(max_length=250)
    slug = models.SlugField()
    description = models.TextField(verbose_name=("A description of what this "
        "system does, and hints on the objective of the optimization."),
        unique=True, blank=False, null=False, default="The aim of this ...")
    is_active = models.BooleanField(default=False, help_text=("If False, then "
            "this system will not be usable."))

    #image_description = models.ImageField()
    level = models.PositiveSmallIntegerField(verbose_name=("Skill level "
                                                           "required by user"),
                                     blank=False, null=False, default=0)
    source = models.TextField(verbose_name=("Python source code that will be "
                                           "executed. A function with the "
                                           "name ``simulate(...)`` must exist. "
                                           "The NumPy library is available "
                                           "as ``np``."),
                              default=u"def simulate(A, B, ):\n    # Code here",
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
    delay_result = models.IntegerField(verbose_name=("Number of seconds before "
                                                     "the result may be shown "
                                                     "to users."), default=0)
    tags = models.ManyToManyField('rsm.Tag')
    known_peak_inputs = models.TextField(verbose_name=("JSON structure giving "
                                                       "the input(s) known to "
                                                       "produce a maximum"),
                                         blank=True)
    cost_per_experiment = models.FloatField(help_text="Dollar cost per run",
                                            default=10.00)
    max_experiments_allowed = models.PositiveIntegerField(default=100)


    #noise_standard_deviation = models.FloatField(default=0,
    #    verbose_name=("Standard deviation of normally distributed noise to add. "
    #        "Both normally and uniformly distributed noise will be added, "
    #        "if specified as non-zero values here."))
    #noise_uniform_multiplier = models.FloatField(default=0,
    #    verbose_name=("Multiplier for uniformally distributed noise: y = mx + "
    #                  "c; this is for multiplier 'm'."))
    #noise_uniform_offset = models.FloatField(default=0,
    #    verbose_name=("Offset for uniformally distributed noise: y = mx + c; "
    #                  "this is for offset value 'c'."))



    def __str__(self):
        return self.full_name


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

