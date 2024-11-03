Extension support
#################

Aioway should be extensible, and provide plugins.

This means that aioway should have its business logic separate,
and expose hooks to strategic locations of its operation.

I'm not sure if this is the best way to extend aioway yet,
because hooks implies that we need to commit to some fixed architecture,
but so far this seems the most promising approach.

File formats
************

Different file formats like audio, video etc should be treated as a core extension.
While aioway is (mostly) a machine learning platform that automatically trains machine learning models,
these extensions would make it possible for aioway to also handle more loosely formatted data,
other than just being able to handle tensors.

At its core though, the loose (file) data formats should be converted into some tensors,
to ensure efficient processing and use with machine learning models.

Non tensors
***********

While aioway, at its core, is a tensor computation engine,
I think that support for non-tensors, and allowing mixing of different datatypes,
is the ultimate spirit of the project.

This means that I should support seamless integration between tensors and non-tensors,
for example allowing integration of LLM into the project (which does not rely on tensors at all),
and design a way for both tensor and non-tensor to coexist and convert to each other.

Perhaps an auto-conversion scheme in the **Must mix different backends** section can be helpful,
if we have a backend for python objects.
After all, if we go back to the fundamentals,
and think of the necessary computation needed for this to happen,
conversion is unavoidable and we would never hope to perform training on python literals for example.
This means that we can isolate different parts of the graphs and run those with different backends.
