from traitlets.config import Config
c = Config()
# show Python tracebacks in the browser
c.VoilaConfiguration.show_tracebacks = True
# also turn on debug logging
c.VoilaConfiguration.debug = True
