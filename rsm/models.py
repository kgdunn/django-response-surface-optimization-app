from django.db import models
import numpy as np


class Person(models.Model):
    """ Defines a person / course participant """
    name = models.CharField(max_length=200, verbose_name="Leaderboard name")
    level = models.SmallIntegerField(verbose_name="Skill level of the user",
                                     blank=False, null=False, default=0)
    email = models.EmailField()

class Token(models.Model):
    """ Tokens capture time/date and permissions of a user to access the
    ``System`` models.
    """
    person = models.ForeignKey('rsm.Person')
    system = models.ForeignKey('rsm.System')
    was_used = models.BooleanField(default=False)
    time_used = models.DateTimeField()
    ip_address = models.GenericIPAddressField(verbose_name=None, name=None,
                                             protocol='both',
                                             unpack_ipv4=False)

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
    token = models.ForeignKey('rsm.Token')
    system = models.ForeignKey('rsm.System')
    # True if the result is successfully simulated (i.e. if the simulation
    # did not time out, or crash for some reason.)
    is_valid = models.BooleanField()
    time_to_solve = models.FloatField(verbose_name="Time to solve model",
                                      blank=False, null=False, default=0.0)
    earliest_to_show = models.DateTimeField(
        verbose_name="Don't show the result before this point in time")

    inputs = models.TextField(verbose_name=("The system inputs logged in JSON "
                                            "format"))
    main_result = models.FloatField(verbose_name="Primary numeric output",
                                   default=-987654321.0)
    other_outputs = models.TextField(verbose_name=("Other outputs produced, "
                                                   "including string messages, "
                                                   "in JSON format, as defined "
                                                   "by the ``System``."),
                                                   blank=True, null=True)

class System(models.Model):
    """ A simulated system, or process. """
    full_name = models.CharField(max_length=250)
    slug = models.SlugField()
    description = models.TextField(verbose_name=("A description of what this "
        "system does, and hints on the objective of the optimization."),
        unique=True, blank=False, null=False, default="The aim of this ...")

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
        blank=True, help_text=('For example: {"water": "-1", "vinegar": "+1"} '
                               'would map the "water" level to -1 and "vinegar" '
                               'level to +1 in the simulation. Leave blank for '
                               'continuous variables.'))
    lower_bound = models.FloatField(blank=True, help_text=("If supplied, will "
        "ensure the user does not enter a value below this."), null=True)
    upper_bound = models.FloatField(blank=True, help_text=("If supplied, will "
            "ensure the user does not enter a value above this."),
                                    null=True)
    default_value = models.FloatField(help_text=("The default used, e.g. in a "
                                                 "multidimensional (>3) plot."))
    units_prefix = models.CharField(max_length=100, help_text=("The prefix for "
            "the units of the input (can be blank)"), blank=True, null=True)
    units_suffix = models.CharField(max_length=100, help_text=("The suffix for "
            "the units of the input (can be blank)"), blank=True, null=True)
    error_message = models.CharField(max_length=200, blank=True,
                                     help_text=("Any error message text that "
                                                "should be shown during input "
                                                "validation."))
    n_decimals = models.PositiveSmallIntegerField(help_text=("The number of "
                            "decimals to show in the numeric representation."),
                                          verbose_name="Number of decimals",
                                                        default=0)



    def __str__(self):
        return self.system.full_name + "::" + self.display_name

