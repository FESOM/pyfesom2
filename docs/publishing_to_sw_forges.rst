Publishing to PyPI and Conda-Forge
==================================

This guide describes how to publish your Python package to PyPI and Conda-Forge using a
CI workflow. The process is automated and only requires pushing a tagged version to your repository.

1. You should consult the ``.github/workflows/pyfesom2-publish.yml`` file in this repository for the full example of the GitHub actions workflow.

2. **Push a tagged version** to trigger the workflow:
   .. code-block:: bash

      git tag v1.0.0
      git push origin v1.0.0

The CI workflow will automatically build and publish your package to both PyPI and Conda-Forge when a tagged version is pushed.

.. note:: The ``ANACONDA_TOKEN`` secret is set on the GitHub organization settings by @pgierz, and expires 31-12-2026!

