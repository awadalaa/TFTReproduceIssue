import json
class JsonCoder(object):
  def encode(self, x):
      return json.dumps(x)

  def decode(self, x):
      return json.loads(x)
