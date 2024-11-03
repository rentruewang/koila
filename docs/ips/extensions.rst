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
