from django.template.defaulttags import register

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter
def get_floatformat(number, decimals):
    return ("{:." + str(int(decimals)) + 'f}').format(number)