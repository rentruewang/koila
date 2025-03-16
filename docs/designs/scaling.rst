Scaling
#######

Horizontal vs vertical
**********************

Consider horizontal distribution of tasks vs vertical scaling.

In a database where one unique query is executed once,
only horizontal distribution is needed (in 1 stage),
whereas vertical distribution can be treated as clones of the current stage (every stage has the same hardware).

However, since aioway is a streaming engine,
vertical distribution (task dependency) can utilize different hardware,
and has to do so while the previous stage is running simultaneously,
so distribution of resources is no longer along the time axis.

Perhaps a pipeline / spark's DAG like architecture can help in vertically scale the application.

Hierarchical RL and world models
********************************

Learnable engines implies engines that learns the outside world on its own
(more reinforcement learning rather than supervised learning),
utilizing goal oriented approaches.

Dynamic imperative API, combined with learnable engines,
would allow option networks and hierarchical RL to be implemented.
Additionally, this can be used for JIT complication.

Engines should be configured to be able to nested hierarchically.
This would help realize hierarchical RL as it would allow for detached graph + world models.

This can do something like self scaling (dynamically inject new pathways),
and achieve arbitrary depth in the hierarchy.

Client server architecture
**************************

A client server architecture would allow the scaling to be handled server side.

Explainable AI
**************

If we keep each and every individual component small, then AI can inherently be explained.

Optionally, explaining by corresponding data is also a great option.

Or, explain
