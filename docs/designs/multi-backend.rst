Multi-Backend Architecture
##########################

Raison D'etre
*************

Currently on the market, there has already been a plethora of "streaming engines".
Be it spark, storm, or other dag runners like airflow.

Considering that those backends already provides a lot of features,
as well as scaling capabilities,
it would be silly not to leverage those resources.

Spark, especially, seems to provide a lot of feature while being very mature and stable.

Why not commit to one
*********************

As of right now, I am unsure which backend is the best.

    In fact, no backend is the best accross the board.

For example, if I want some local experiement, running Spark seems overkill,
but it would be amazing in an enterprise setting when lots of data is coming in.
Not knowing which backend to commit to,
I think it is a smart choice to make the architecture agnostic to different backends.

Should I mix different backend
******************************

No, backend mixing can lead to difficult situation,
where you have to pay a big price for serialization / deserialization,
as different backends uses different data formats and evaluation strategies.

For example,
spark uses a lazy evaluation approach, and can perform query optimization.
All of which is nullified if it has to synchronize with, say, a local numpy backend.

**Therefore, what I'm doing here with the backends
is closer to applying strategy pattern to all the functionality the platform provides.**

Must mix different backends
***************************

If I must mix different backends, there are a couple of ways to achieve it.

#. Every step in the pipeline can use a different backend.
#. Only across *blocks* of code you can switch to a different backend.

These 2 conditions are fundamentally very similar,
the only difference being the frequency of the switch.

I believe that the second one is superior.

The first one would have the following options::

    #. Serialize at every single step, use a common format across all backends.
    #. Use dynamic programming and serialize only when necessary.

Needless to say, the second option is better and work with the *block* strategy as well.

However, the block strategy as another advantage,
which happens when only 1 backend is used.
In which case, we do not even need to serialize at all.

Data formats
************

Considering that computation is intrinsically tied to the data on which they operate,
different backends would require the use of different data.

However, since computation themselves should be backend agnostic,
this creates a difficult situation of how to design a good API,
s.t. computation can be accessed globally and locally (on data themselves).

A good approach is to define all of the computation on the object itself, like spark,
and have the global functions call the member functions (like torch and numpy).
