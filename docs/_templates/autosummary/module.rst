{% if modules %}
{# This is a package with submodules #}
{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

.. rubric:: Modules

.. autosummary::
   :toctree: .
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}

{% else %}
{# This is a regular module with code #}
{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}

   {% if classes %}
   .. rubric:: Classes

   {% for item in classes %}
   .. dropdown:: {{ item }}

      .. autoclass:: {{ fullname }}.{{ item }}
         :members:
   {%- endfor %}
   {% endif %}

   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
{% endif %}
