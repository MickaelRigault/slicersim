{% if fullname == 'slicersim' %}
slicersim
=========

.. automodule:: slicersim

.. rubric:: Modules

.. autosummary::
   :toctree: .
   :recursive:

   slicersim.detector
   slicersim.extra
   slicersim.iotools
   slicersim.lazuli
   slicersim.nea
   slicersim.profiles
   slicersim.scene
   slicersim.simulation
   slicersim.spectrograph
   slicersim.study
   slicersim.telescope
   slicersim.thermal
   slicersim.utils

{% else %}
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