def pytest_configure(config):
    plugin = config.pluginmanager.getplugin("mypy")
    plugin.mypy_argv.extend(["--config-file", "./test_config/mypy.ini"])
