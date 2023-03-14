import setuptools

setuptools.setup(
    name="artist_style",
    url="https://github.com/skarlett992/artist_style",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.1',
        'tensorboard>=2.3.0',
        'Pillow>=7.0.0'
    ],
    entry_points={
        "console_scripts": [
            "style-transfer = style_transfer.__main__:main"
        ]
    }
)