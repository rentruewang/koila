Miscellaneous
#############

Presentation
************

.. todo::

    More professional looking flowchart with either graphviz or diagrams.

User interface
**************

.. todo::

    Would need a user interface to interact with the application.

Language
********

.. todo::

    Grammar redesigned to be close to SQL.

    A graph-based language like Cypher should be supported too.

Relational algebra
******************

.. todo::

    Does it make sense to keep extending the amount of relational algebra operators?

    If so, where do we associate it with different backends?

Columns or Tables
*****************

A big part of the compiler of aioway is solving how the higher level constraints can directly be mapped to different models. This imposes 2 challenges:

#. How to ensure that models are registered in an extensible way.
#. How to associate those constraints to the physical properties of the different models.

For this, it becomes very important what basic atomic unit the compiler wants to use, tables or columns.
