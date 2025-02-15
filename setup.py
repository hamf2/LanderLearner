from setuptools import setup, find_packages

setup(
    name="lander_learner",
    version="0.1.0",
    description="A 2D Lunar Lander simulation RL demo project",
    author="Harry Fieldhouse",
    author_email="harryamfieldhouse@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "lunar_lander": ["scenarios/*.json", "assets/*.png"],
    },
    install_requires=[
        "gymnasium==1.0.0",
        "stable-baselines3[extra]==2.5.0",
        "pymunk==6.11.1",
        "pygame==2.6.1",
        "numpy==2.2.2"
    ],
    entry_points={
        "console_scripts": [
            "lander_learner=lunar_lander.main:main",
        ],
    },
)