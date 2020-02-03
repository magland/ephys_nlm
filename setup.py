import setuptools

pkg_name = "ephys_nlm"

setuptools.setup(
    name=pkg_name,
    version="0.1.1",
    author="Jeremy Magland",
    author_email="jmagland@flatironinstitute.org",
    description="Non-local means denoising of multi-channel electrophysiology timeseries.",
    packages=setuptools.find_packages(),
    scripts=[],
    install_requires=[
        # Intentionally not including numpy and pytorch in this list -- they should be installed separately
        'spikeextractors'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
