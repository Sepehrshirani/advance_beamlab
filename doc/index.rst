:html_theme.sidebar_secondary.remove: true

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :end-before: <!-- doc-split: background -->

Where to go next
================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Examples
      :link: auto_examples/index
      :link-type: doc

      Seven worked examples, from the correlated-source problem in isolation
      through to real MEG recordings. Every figure is produced by the code on
      the page.

   .. grid-item-card:: Mathematical background
      :link: background
      :link-type: doc

      Every equation derived from first principles, so each method can be
      understood and tuned without reading the papers first.

   .. grid-item-card:: Parameter tuning
      :link: tuning
      :link-type: doc

      What each parameter controls and what changes when you move it, per
      algorithm.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc

      Every public function and class, documented from its in-code docstring.

.. toctree::
   :hidden:
   :maxdepth: 2

   Examples <auto_examples/index>
   Mathematical background <background>
   Parameter tuning <tuning>
   API reference <api>
