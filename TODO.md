1. Add more models
   - LapSRN
2. Add Generative train state
3. Add Coordinate-based train state(LIIF)
   - But jax.scipy's map-coordinate does not work same as pytorch's F.grid_sample.
4. Add BSRGAN's preprocessing function. But it is hard to compile it to jax.numpy.
5. Add Module Testing Codes.
6. Maybe it is better to integrate loss weight into loss class. 