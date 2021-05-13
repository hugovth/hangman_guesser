# common-config

Contains common configuration for repositories. This config can be used via git submodules.

More info on the [wiki](http://wiki.getdashmote.com/tech/python-development-guide#common-config).

## Usage

> **Note**: normally, you would not have to initialize a repo, as either your repository already has a .common-config, or when creating a new repo, the [cookiecutter](https://github.com/dashmote/cookiecutter-template) provides it.

To initialize a repo with the configuration in this repo.

```bash
cd path/to/my-repo
path/to/common-config/init.sh
```

The repo 'my-repo' should now have all the configuration files as symlinks to common-config that is now a git submodule in my-repo under `.common-config/`.

### Deinitialize

For whatever reason, if you wish to deinitialize, i.e. reverse the process of initializiation:

```bash
cd path/to/my-repo
path/to/common-config/deinit.sh
```
