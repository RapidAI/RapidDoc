from pathlib import Path
from setuptools import setup, find_packages
from kitty_doc.version import __version__


def parse_requirements(filename):
    with open(filename) as f:
        lines = f.read().splitlines()

    requires = []

    for line in lines:
        if "http" in line:
            pkg_name_without_url = line.split('@')[0].strip()
            requires.append(pkg_name_without_url)
        else:
            requires.append(line)

    return requires


if __name__ == '__main__':
    with Path(Path(__file__).parent,
              'README.md').open(encoding='utf-8') as file:
        long_description = file.read()
    setup(
        name="kitty_doc",  # 项目名
        version=__version__,  # 自动从tag中获取版本号
        license="Apache 2.0",
        author='',  # 作者名
        author_email='',  # 作者邮箱
        packages=find_packages() + ["kitty_doc.resources"],  # 包含所有的包
        package_data={
            "kitty_doc.resources": ["**"],  # 包含 kitty_doc/resources 目录下的所有文件
            "": ["*.yaml"],  # 包含所有包里的 .yaml 文件
        },
        install_requires=parse_requirements('requirements.txt'),  # 项目依赖的第三方库
        # extras_require={
        # },
        description="A practical tool for converting PDF to Markdown",  # 简短描述
        long_description=long_description,  # 详细描述
        long_description_content_type="text/markdown",  # 如果README是Markdown格式
        project_urls={
            "Home": "https://github.com/hzkitty",
            "Repository": "https://github.com/hzkitty/KittyDoc",
        },
        keywords=["kitty-doc, kitty_doc, onnx, convert, pdf, markdown"],
        classifiers=[
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
        ],
        python_requires=">=3.10,<3.14",  # 项目依赖的 Python 版本
        entry_points={
            "console_scripts": [

            ],
        },  # 项目提供的可执行命令
        include_package_data=True,  # 是否包含非代码文件，如数据文件、配置文件等
        zip_safe=False,  # 是否使用 zip 文件格式打包，一般设为 False
    )