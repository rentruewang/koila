Schemas
#######

Schemas are the data types in the project.
It tells the pipeline, before having to perform any heavy computation,
what kinds of data would be allowed.
Then, the pipeline can decide how to process it accordingly.

``TableAttrSet`` and ``ColumnAttr``
************************************

``TableAttrSet`` shows the schema of an entire table, vs ``ColumnAttr``'s description of a column.

A ``ColumnAttr`` can be used to initialize a ``TableAttrSet``,
which is a mapping of string to ``ColumnAttr``.

Note that ``Attr``s are not ``Schema``s, the latter is used for integrating with external DBs.

The benefit of separating ``Attr``s and ``Schema``s is that
we are able to ``encode`` data of schema to ``TableAttrSet`` schema,
which splits the internal and external data format handling,
allowing a wider range of support.
