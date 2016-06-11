UE = require("./exception")


class Vector
  constructor: (n, elements) ->
    if elements instanceof Array
      throw new UE.UserException("wrong number of elements when construct Vector object") if elements.length isnt n
      @array = elements
    else
      @array = new Array(n)
    @length = n

  innerProduct: (v) ->
    if v.length isnt @length
      throw new UE.UserException("different dimensions' vectors can't product")
    55

  setDevice: ->



exports.Vector = Vector