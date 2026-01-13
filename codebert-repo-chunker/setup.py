from setuptools import setup, find_packages

setup(
    name="codebert-repo-chunker",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "chunk-repo=scripts.process_repo:main",
            "analyze-repo=scripts.analyze_repo:main",
        ],
    },
)