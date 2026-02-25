{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% block modules %}
{% if modules %}
.. autosummary::
   :toctree:
   :recursive:

{% for item in modules %}
{% if not item.split('.')[-1].startswith('tests') %}
   {{ item }}
{% endif %}
{% endfor %}

{% endif %}
{% endblock %}