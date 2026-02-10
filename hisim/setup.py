from setuptools import setup, find_packages


def get_version():
    version = "0.1.0"
    with open("src/hisim/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip(' \n"')
    return version


setup(
    name="hisim",
    version=get_version(),
    url="https://github.com/alibaba/tair-kvcache.git",
    description="A High-Fidelity LLM inference simulator ",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
