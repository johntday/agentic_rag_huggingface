To set the level on root explicitly do `logging.getLogger().setLevel(logging.DEBUG)`. But ensure you've called `basicConfig()` before hand so the root logger initially has some setup. I.e.:

```
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger('foo').debug('bah')
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('foo').debug('bah')
```
Also note that "Loggers" and their "Handlers" both have distinct independent log levels. So if you've previously explicitly loaded some complex logger config in you Python script, and that has messed with the root logger's handler(s), then this can have an effect, and just changing the loggers log level with `logging.getLogger().setLevel(..)` may not work. This is because the attached handler may have a log level set independently. This is unlikely to be the case and not something you'd normally have to worry about.

https://stackoverflow.com/questions/38537905/set-logging-levels

https://docs.python.org/3/howto/logging.html

