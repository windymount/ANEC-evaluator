from setuptools import setup, find_packages

setup(
    name="anec_evaluator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "numpy",
    ],
    entry_points={
        'console_scripts': [
            'get_anec=anec_evaluator.cli:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for evaluating ANEC (Automatic Neural Concept Extraction)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ANEC-evaluator",
) 