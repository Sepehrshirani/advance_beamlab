:html_theme.sidebar_secondary.remove: true

.. include:: ../README.md
   :parser: myst_parser.sphinx_
   :end-before: <!-- doc-split: background -->

Where to go next
================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Which method should I use?
      :link: choosing
      :link-type: doc

      A decision guide keyed on the data you have, with what this package
      actually demonstrates for each method, including the cases where a
      plain LCMV is the better answer.

   .. grid-item-card:: What a constraint does
      :link: panel
      :link-type: doc

      An interactive panel. Place two correlated sources on a cortex and watch
      the constraint table while you move the correlation: LCMV's off-diagonal
      gain runs away and takes the amplitude with it, MCMV's stays pinned at
      zero.

   .. grid-item-card:: Examples
      :link: auto_examples/index
      :link-type: doc

      Nine worked examples, from the correlated-source problem in isolation
      through to real MEG recordings and a finite-element EEG head model. Every
      figure is produced by the code on the page.

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

   Which method should I use? <choosing>
   What a constraint does <panel>
   Examples <auto_examples/index>
   Mathematical background <background>
   Parameter tuning <tuning>
   API reference <api>
