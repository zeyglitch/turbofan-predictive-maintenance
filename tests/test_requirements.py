"""The short lists of the Docker images pin the same versions as requirements.txt."""

import pytest

from turbofan.config import ROOT


def pinned(file_name: str) -> dict[str, str]:
    """{package: version} for every `package==version` line of a requirements file."""
    lines = (ROOT / file_name).read_text(encoding="utf-8").splitlines()
    pairs = [line.split("==") for line in lines if "==" in line and not line.startswith("#")]
    return {name.strip(): version.strip() for name, version in pairs}


@pytest.mark.parametrize("file_name", ["requirements-api.txt", "requirements-dashboard.txt"])
def test_image_requirements_match_the_main_list(file_name):
    main = pinned("requirements.txt")
    image = pinned(file_name)
    assert image  # the file is not empty
    for name, version in image.items():
        assert main.get(name) == version, (
            f"{name} is {version} in {file_name}, {main.get(name)} in requirements.txt"
        )


def test_the_api_image_does_without_the_heavy_tools():
    image = pinned("requirements-api.txt")
    assert not {"mlflow", "shap", "streamlit", "dvc"} & set(image)


def test_the_dashboard_image_does_without_torch():
    image = pinned("requirements-dashboard.txt")
    assert not {"torch", "mlflow", "shap", "dvc"} & set(image)
