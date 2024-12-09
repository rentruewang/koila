Scope resolution
################

Different programming languages / possible frontends have different name resolution rules.

Relational algebra, on the other hand, the columns are tied to the table inherently.

The table way is really easy to implement, but inherently less flexible.

How to achieve both
*******************

There are a couple of options to implement this.

#. Do not tie the name resolution to the tables themselves. That is to say, to use an additional resolver. However, this complexity comes at a cost and it is really difficult to maintain and relatively difficult to debug.
#. Have resolvers tied to the tables themselves, so that they can use different strategies to resolve for available variables.
#. Possibly the best way, is to not worry about this at all. Considering that the data frame abstraction follows closely to how streaming works, frontend should have their own name resolution and enforce the relational algebra column resolution.
