from setuptools import setup

setup(
    name='lmb',
    version='0.1',
    author = "Jonatan Olofsson",
    author_email = "jonatan.olofsson@gmail.com",
    description = (""),
    license = "GPLv3",
    keywords = "multi-target tracking multi-bernoulli",
    url = "http://github.com/jonataolofsson/lmb.git",
    packages=['lmb'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'lapjv', 'shapely']
)
