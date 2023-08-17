import codecs
import os
import re
import sys

from setuptools import find_packages, setup

NAME = "pspline_psd"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join("src", "pspline_psd", "__init__.py")
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.8",
]

# ensure python 3.8
python_version = (3, 8)
if sys.version_info == python_version:
    error = """
    This package requires Python {}.{}
    """.format(
        *python_version
    )
    sys.exit(error)


INSTALL_REQUIRES = [
    "arviz",
    "scikit-fda",
    "matplotlib",
    "loguru",
    "bilby",
    "statsmodels",

]
EXTRA_REQUIRE = {
    "dev": [
        "pytest>=7.2.2",
        "pytest-cov>=4.1.0",
        "pre-commit",
        "flake8>=5.0.4",
        "black>=22.12.0",
        "jupyter-book",
    ]
}

HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        use_scm_version={
            "write_to": os.path.join("src", NAME, "{0}_version.py".format(NAME)),
            "write_to_template": '__version__ = "{version}"\n',
        },
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        url=find_meta("uri"),
        license=find_meta("license"),
        description=find_meta("description"),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_data={
            NAME: [
                "plotting/style.mlpstyle",
            ]
        },
        package_dir={"": "src"},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRE,
        classifiers=CLASSIFIERS,
        zip_safe=True,
        entry_points={"console_scripts": []},
    )
