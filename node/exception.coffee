
class UserException
  constructor: (msg) ->
    @message = msg
    @name = "User Error"

  toString: ->
    "#{@name}: \"#{@name}\""

exports.UserException = UserException