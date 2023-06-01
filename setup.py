import setuptools


setuptools.setup(
    name="flaxsr",
    version="0.0.4",
    author="dslisleedh",
    author_email="dslisleedh@gmail.com",
    description="Super Resolution models with Jax/Flax",
    long_description=open('README.md', 'rt').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dslisleedh/FlaxSR",
    project_urls={
        "Bug Tracker": "https://github.com/dslisleedh/FlaxSR/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing"
    ],
    install_requires=[
        "jax",
        "jaxlib",
        "flax",
        "einops",
        "tensorflow",
        "numpy",
        "optax"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
