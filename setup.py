from setuptools import setup, find_packages


try:
    with open('README.md') as f:
        readme = f.read()
except IOError:
    readme = ''


name = 'pdf-struct'
exec(open('pdf_struct/_version.py').read())
release = __version__
version = '.'.join(release.split('.')[:2])

with open('requirements.txt') as fin:
    requirements = [line.strip() for line in fin]

setup(
    name=name,
    author='Yuta Koreeda',
    author_email='yuta.koreeda@hal.hitachi.com',
    maintainer='Yuta Koreeda',
    maintainer_email='yuta.koreeda@hal.hitachi.com',
    version=release,
    description='Logical structure analysis of visually structured documents.',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/stanfordnlp/pdf-struct',
    packages=find_packages(),
    install_requires=requirements,
    setup_requires=requirements,
    license='Apache',
    entry_points = {
        'console_scripts': ['pdf-struct=pdf_struct.cli:cli'],
    },
    classifiers=[
        "Environment :: Console",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)
