from django.db import models
import numpy as np

# Person
class Person(models.Model):
    """ Defines a person / course participant """
    name = models.CharField(max_length=200)
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
    """Tags for ``Systems``."""
    short_name = models.CharField(max_length=50)
    description = models.CharField(max_length=150)
    
    def __str__(self):
        return self.short_name
    
    
class Result(models.Model):
    person = models.ForeignKey('rsm.Person')
    token = models.ForeignKey('rsm.Token')
    system = models.ForeignKey('rsm.System')
    is_valid = models.BooleanField()
    time_to_solve = models.FloatField(verbose_name="Time to solve model",
                                      blank=False, null=False, default=0.0)
    earliest_to_show = models.DateTimeField(
        verbose_name="Don't show the result before this point in time")
  
    main_result = models.FloatField(verbose_name="Primary numeric output", 
                                   default=-987654321.0)
    other_outputs = models.TextField(verbose_name=("Other outputs produced, "
                                                   "including string messages, " 
                                                   "in JSON format, as defined " 
                                                   "by the ``System``."), 
                                                   blank=True, null=True)
      
class System(models.Model):
    full_name = models.CharField(max_length=250)
    description = models.TextField(verbose_name=("A description of what this "
        "system does, and hints on the objective of the optimization."), 
        unique=True, blank=False, null=False, default="The aim of this ...")
    
    #image_description = models.ImageField()    
    level = models.PositiveSmallIntegerField(verbose_name=("Skill level "
                                                           "required by user"),
                                     blank=False, null=False, default=0)
    source = models.TextField(verbose_name=("Python source code that will be "
                                           "executed. Called function with " 
                                           "name ``simulate(...)`` must exist"),
                              default=u"def simulate(A, B, ):\n    # Code here",
                              unique=True, blank=False)
    simulation_timeout = models.PositiveSmallIntegerField(blank=False, 
                                                          null=False,
                                                          default=0,
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
    noise_standard_deviation = models.FloatField(default=0,
        verbose_name=("Amount of normally distributed noise to add. Both "
                      "normally and uniformly distributed noise will be added, "
                      "if specified as non-zero values here."))
    noise_uniform_multiplier = models.FloatField(default=0,
        verbose_name=("Multiplier for uniformally distributed noise: y = mx + "
                      "c; this is for multiplier 'm'."))
    noise_uniform_offset = models.FloatField(default=0,
        verbose_name=("Offset for uniformally distributed noise: y = mx + c; "
                      "this is for offset value 'c'."))
    
    
    def __str__(self):
        return self.full_name