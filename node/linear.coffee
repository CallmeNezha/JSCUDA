UE = require("./exception")
JC = require("./jc/build/Debug/jc.node")

class VectorD
  constructor: ( n, elements = undefined ) ->
    @elements = undefined
    @length   = Math.ceil( n )

    # Filter parameter
    if @length < 1 then throw new UE.UserException( "'n' \<uint32\> must greater than zero" )

    if elements instanceof JC.DeviceFloat32Array
      if elements.length isnt @length then throw new UE.UserException( "Wrong number of elements when construct VectorD object" )
      @elements = elements
    else if elements instanceof Float32Array
      @elements = new JC.DeviceFloat32Array( @length )
      @copyFrom( @length, elements )
    else
      @elements = new JC.DeviceFloat32Array( @length )

    # Check before return
    if !( @elements instanceof JC.DeviceFloat32Array ) then throw new UE.UserException( "VectorD object construct failed" )

  # Explicitly reclaim device memory , cudaFree under the hood
  destroy: ->
    @elements.destroy()
    @length = 0

  copy: ( v ) ->
    if @length != v.length then throw new UE.UserException( "'v.length' mismatch" )
    @elements.copy( v.elements, 0, 0, @length )



  copyFrom: ( n, array ) ->
    n = Math.ceil( n )
    if n < 1 then throw new UE.UserException( "'n' \<uint32\> must  greater than zero" )
    if !( array instanceof Float32Array ) then throw new UE.UserException( "'array' must be Float32Array" )
    if n > array.length or n > @elements.length then throw new UE.UserException( "'n' exceed range of array" )
    # DeviceFloat32Array::copyFrom( host, offset_h, offset_d, size )
    @elements.copyFrom( array, 0, 0, n )

  copyTo: ( n, array ) ->
    n = Math.ceil( n )
    if n < 1 then throw new UE.UserException( "'n' \<uint32\> must  greater than zero" )
    if !( array instanceof Float32Array ) then throw new UE.UserException( "'array' must be Float32Array" )
    if n > array.length or n > @elements.length then throw new UE.UserException( "'n' exceed range of array" )
    # DeviceFloat32Array::copyTo( host, offset_h, offset_d, size )
    @elements.copyTo( array, 0, 0, n )


  add: ( v ) ->
    if !( v instanceof VectorD && v.elements? ) then throw new UE.UserException( "'v' must be VectorD" )
    if v.length != @length then throw new UE.UserException( "'v.length' mismatch" )
    JC.vectorAdd( @, v )


class MatrixD
  constructor:( m, n, elements = undefined ) ->
    @numRow = m
    @numCol = n

    if elements instanceof JC.DeviceFloat32Array
      if elements.length isnt m * n then throw new UE.UserException( "wrong number of elements when construct Matrix object" )
      @elements = elements
    else
      @elements = new Float32Array( m * n )




module.exports = JC
exports = module.exports
exports.VectorD = VectorD
exports.MatrixD = MatrixD