from django.template.defaulttags import register


@register.filter
def multiply(value1, value2):
    return value1 * value2

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter
def dict_iteration(dictionary, key):
    """ This is required because the dictionary is passed in as a variable,
    and while the regular Django templates have a dictionary iterator, you
    have to specify the dictionary name in the template. We can't do that! """
    return [item for item in dictionary.get(key).iteritems()]

@register.filter
def get_floatformat(number, decimals):
    return ("{:." + str(int(decimals)) + 'f}').format(number)