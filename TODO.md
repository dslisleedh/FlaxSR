1. Add more models
   - LapSRN
2. Add Generative train state
3. Add Coordinate-based train state(LIIF)
   - But jax.scipy's map-coordinate does not work same as pytorch's F.grid_sample.
4. Maybe add preprocessing steps into train_step
5. Maybe integrating training function into model is more flexible to implement?
6. Add BSRGAN's preprocessing function. But it is hard to compile it to jax.numpy.
7. Add Module Testing Codes.
