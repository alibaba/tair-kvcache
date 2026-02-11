from setuptools import setup, find_packages

setup(
    name='kvcm-trace-converter',
    version='1.0.0',
    description='Trace format converter for KVCacheManager Optimizer',
    author='KVCacheManager Team',
    packages=find_packages(),
    install_requires=[
        'transformers>=4.30.0',
        'tqdm>=4.65.0',
        'torch>=2.0.0',
    ],
    entry_points={
        'console_scripts': [
            'trace-converter=trace_converter:main',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
