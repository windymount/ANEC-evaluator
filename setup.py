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
    author="Ge Yang",
    author_email="geyan@ucsd.edu",
    description="A tool for evaluating ANEC (Accuracy under specified Number of Effective Concepts)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/windymount/ANEC-evaluator",
) 