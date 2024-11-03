Training
########

Offline training
****************

.. todo::

    Training models are very important.
    Is there a good way for people to train models without having to know what goes on under the hood?

    For example: pytorch would require people to call the following sequence of steps

    .. code-block:: python

        loss = ... # Something you compute, must be a scalar.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Offline models don't necessarily interact with the environment,
    which means the policy is not collecting data in the environment.

Online training
***************

.. todo::

    Online training refers to the type of training that updates on the fly,
    rather than having different stages of training and inference.

    Online models interact with the environment, which means the collecting policy is the one trained. This means Trainer can be extended to perform updates in real time on a Model. No issues here.

    Capabilities to support of RL models in the future is crucial.
