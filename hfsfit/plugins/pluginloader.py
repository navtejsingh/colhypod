# pluginloader.py

import imp
import os
from hfsfit import plugin_dir

MainModule = "__init__"

def getPlugins():
    plugins = {}
    possibleplugins = os.listdir(plugin_dir)
    for i in possibleplugins:
        location = os.path.join(plugin_dir, i)
        if not os.path.isdir(location) or not MainModule + ".py" in os.listdir(location):
            continue
        info = imp.find_module(MainModule, [location])
        plugins[i] = info
        #plugins.append({"name": i, "info": info})
    return plugins

def loadPlugin(plugin):
    return imp.load_module(MainModule, *plugin)