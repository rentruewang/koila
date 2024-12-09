Imperative API
##############

Even though aioway is a streaming engine,
for which relation algebra is best used to describe the data flow when inputs are ticks,
there are times where we really want to provide an imperative API,
because it's just more natural, and we give instructions in an imperative manner.

Streaming
*********

Aioway is a streaming engine built for training and inference on streams.

However, imperative API naturally assumes a bounded table because we are only focusing on 1 instance.

While it doesn't really make sense to give imperative instructions in a non-streaming context,
(because states are implicit in any instructions, which requires consideration for the past),
it might still be a nice way to interact with the underlying infrastructure.

Syntax sugar
************

We could do syntax sugar on top of the streaming engine,
allowing users to use abstract modules (trained and run with the streaming engine),
making the imperative part self contained.

Symbolic on top of non-symbolic
*******************************

Allowing symbolically control non-symbolic features is a gigantic breakthrough,
as it allows deep learning (animal senses) to be controlled by logic.
