# Current challenges and strategies

1. `isinstance(fake, Tensor)` should be true to mimic a pytorch tensor.

   This should be doable. See Python's `unittest.mock`. Strategy: use an `Imposter` meta class.

2. Calculation could be faster.

   Right now it runs in Python. Nothing is stopping me from using C++.

3. Modularization wasn't good in v1 branches.

   Well it was a product of putting the MVP up there. So the quality should be improved.

4. More documentation.

   Previously it's pretty lacking because I didn't have time to put it on.
