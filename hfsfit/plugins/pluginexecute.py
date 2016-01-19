
import os
import pluginloader
from hfsfit import data_dir

#for i in pluginloader.getPlugins():
#    print i
#    print("Loading plugin " + i["name"])

plugins = pluginloader.getPlugins()

if plugins.has_key('extract'):
    plugin = pluginloader.loadPlugin(plugins['extract'])
    print plugin.run(os.path.join(data_dir, "64423m.fits"))
else:
    raise KeyError("Plugin extract not found.")

#plugin = pluginloader.loadPlugin("__init__.py", "")
#plugin.run(os.path.join(data_dir, "64423m.fits"))