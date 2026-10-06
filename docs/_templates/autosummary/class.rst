{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :show-inheritance:

{% set public_methods = methods | reject("equalto", "__init__") | list %}
{% if public_methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
{% for item in public_methods %}
      ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}

{% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
{% for item in attributes %}
      ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}

{% for item in public_methods %}
   .. automethod:: {{ item }}
{%- endfor %}

{% for item in attributes %}
   .. autoattribute:: {{ item }}
{%- endfor %}
