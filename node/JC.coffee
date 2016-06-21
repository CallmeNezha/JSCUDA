JC = require("./jc/build/Debug/jc.node")


class UserException
  constructor: (msg) ->
    @message = msg
    @name = "User Error"

  toString: ->
    "#{@name}: \"#{@name}\""

UE = {
  UserException: UserException
}


class VectorD
  constructor: ( n, elements = undefined ) ->
# Read only properties
    @length   = Math.ceil( n )
    # !Read only properties

    # Private member
    @elements = undefined
    # !Private member


    # Filter parameter
    if @length < 1 then throw new UE.UserException( "'n' <uint32> must greater than zero" )

    if elements?
      if elements.length isnt @length then throw new UE.UserException( "'elements''s dimension mismatch" )
      if elements instanceof JC.DeviceFloat32Array
        @elements = elements
      else if elements instanceof Float32Array
        @elements = new JC.DeviceFloat32Array( @length )
        @copyFrom( @length, elements )
    else
      @elements = new JC.DeviceFloat32Array( @length )

    if !(@elements instanceof JC.DeviceFloat32Array) then throw new UE.UserException( "'elements''s type mismatch" )


# Explicitly reclaim device memory , cudaFree under the hood
  destroy: ->
    @elements.destroy()
    @length = 0
    undefined

  copy: ( v ) ->
    if !( v instanceof VectorD and v.elements? ) then throw new UE.UserException( "'v' must be VectorD" )
    if @elements.length isnt v.elements.length then throw new UE.UserException( "'v''s dimension' mismatch" )
    @elements.copy( v.elements, 0, 0, @elements.length )
    @



  copyFrom: ( n, array ) ->
    n = Math.ceil( n )
    if n < 1 then throw new UE.UserException( "'n' <uint32> must  greater than zero" )
    if !( array instanceof Float32Array ) then throw new UE.UserException( "'array' must be Float32Array" )
    if n > array.length or n > @elements.length then throw new UE.UserException( "'n' exceed range of array" )
    # DeviceFloat32Array::copyFrom( host, offset_h, offset_d, size )
    @elements.copyFrom( array, 0, 0, n )
    @

  copyTo: ( n, array ) ->
    n = Math.ceil( n )
    if n < 1 then throw new UE.UserException( "'n' <uint32> must  greater than zero" )
    if !( array instanceof Float32Array ) then throw new UE.UserException( "'array' must be Float32Array" )
    if n > array.length or n > @elements.length then throw new UE.UserException( "'n' exceed range of array" )
    # DeviceFloat32Array::copyTo( host, offset_h, offset_d, size )
    @elements.copyTo( array, 0, 0, n )
    @


  add: ( v ) ->
    if !( v instanceof VectorD and v.elements? ) then throw new UE.UserException( "'v' must be VectorD" )
    if v.length isnt @length then throw new UE.UserException( "'v''s dimension'' mismatch" )
    JC.vectorAdd( @, v )
    @

  dot: ( v ) ->
    if !( v instanceof VectorD and v.elements? ) then throw new UE.UserException( "'v' must be VectorD" )
    if v.length isnt @length then throw new UE.UserException( "'v''s dimension' mismatch" )
    JC.vectorDot( @, v )

  norm: ->
    JC.vectorNorm( @ )

  normSq: ->
    norm = JC.vectorNorm( @ )
    norm * norm

  multiplyScalar: ( s ) ->
    if !( typeof s is 'number' ) then throw new UE.UserException( "'s' must be number" )
    JC.vectorMulScalar( @, s )
    @

  tensor: ( v, m ) ->
    if !( v instanceof VectorD and v.elements? ) then throw new UE.UserException( "'v' must be VectorD" )
    if !( m instanceof MatrixD and m.elements? ) then throw new UE.UserException( "'m' must be MatrixD" )
    if m.numRow isnt @length or m.numCol isnt v.length then throw new UE.UserException( "'m''s dimension mismatch" )
    JC.vectorRank( @, v, m )
    m

  setZero: ->
    @elements.setValue(0)
    @




class MatrixD
  constructor:( m, n, elements = undefined ) ->

# Read only properties
    @numRow = Math.ceil( m )
    @numCol = Math.ceil( n )
    @transposed = false
    # !Read only properties

    # Private member
    @elements = undefined
    # !Private member

    # Filter parameter
    if @numRow < 1 or @numCol < 1 then throw new UE.UserException( "'m, n' <uint32> must greater than zero" )

    if elements?
      if elements.length isnt @numCol * @numRow then throw new UE.UserException( "'elements''s dimension mismatch" )
      if elements instanceof JC.DeviceFloat32Array
        @elements = elements
      else if elements instanceof Float32Array
        @elements = new JC.DeviceFloat32Array( elements.length )
        @copyFrom( elements.length, elements )
    else
      @elements = new JC.DeviceFloat32Array( @numCol * @numRow )

    if !(@elements instanceof JC.DeviceFloat32Array) then throw new UE.UserException( "'elements''s type mismatch" )


# Explicitly reclaim device memory , cudaFree under the hood
  destroy: ->
    @elements.destroy()
    @elements = undefined
    @numRow = 0
    @numCol = 0
    undefined

  T: ->
    t = new MatrixD( @numCol, @numRow, @elements )
    t.transposed = true
    t

  copy: ( m ) ->
    if !( m instanceof MatrixD and m.elements? ) then throw new UE.UserException( "'m' must be MatrixD" )
    if @elements.length isnt m.elements.length or @numRow isnt m.numRow or @numCol isnt m.numCol then throw new UE.UserException( "'m''s dimension mismatch" )

    @elements.copy( m.elements, 0, 0, @elements.length )
    @transposed = m.transposed
    @

# TODO: Array's elements must be aligned with respect to matrix transposed state. For example: row major storage when matrix is transposed
  copyFrom: ( n, array ) ->
    n = Math.ceil( n )
    if n < 1 then throw new UE.UserException( "'n' <uint32> must  greater than zero" )
    if !( array instanceof Float32Array ) then throw new UE.UserException( "'array' must be Float32Array" )
    if n > array.length or n > @elements.length then throw new UE.UserException( "'n' exceed range of array" )
    # DeviceFloat32Array::copyFrom( host, offset_h, offset_d, size )
    @elements.copyFrom( array, 0, 0, n )
    @

# TODO: Array must be realigned after copyTo with respect to matrix transposed state. For example: row major to column major storage when matrix is transposed
  copyTo: ( n, array ) ->
    n = Math.ceil( n )
    if n < 1 then throw new UE.UserException( "'n' <uint32> must  greater than zero" )
    if !( array instanceof Float32Array ) then throw new UE.UserException( "'array' must be Float32Array" )
    if n > array.length or n > @elements.length then throw new UE.UserException( "'n' exceed range of array" )
    # DeviceFloat32Array::copyTo( host, offset_h, offset_d, size )
    @elements.copyTo( array, 0, 0, n )
    @

  multiplyScalar: ( s ) ->
    if !( typeof s is 'number' ) then throw new UE.UserException( "'s' must be number" )
    JC.matrixMulScalar( @, s )
    @

  multiplyMatrix: ( mb, mc ) ->
    if !( mb instanceof MatrixD and mb.elements? and  mc instanceof MatrixD and mc.elements? ) then throw new UE.UserException( "'mb, mc' must be MatrixD" )
    if @numCol isnt mb.numRow or @numRow isnt mc.numRow or mb.numCol isnt mc.numCol then throw new UE.UserException( "'mb, mc''s dimension mismatch" )
    JC.matrixMulMatrix( @, mb, mc )
    mc.transposed = false
    mc

  multiplyVector: ( va , vb ) ->
    if !( va instanceof VectorD and vb instanceof VectorD and va.elements? and vb.elements? ) then throw new UE.UserException( "'va, vb' must be VectorD" )
    if @numCol isnt va.length isnt vb.length then throw new UE.UserException( "'va, vb''s dimension mismatch" )
    JC.matrixMulVector( @, va, vb )
    vb



  add: ( mb, mc ) ->
    if !( mb instanceof MatrixD and mb.elements? ) then throw new UE.UserException( "'mb' must be MatrixD" )
    if !( mc instanceof MatrixD and mc.elements? ) then throw new UE.UserException( "'mc' must be MatrixD" )
    if @numCol isnt mb.numRow or @numRow isnt mc.numRow or mb.numCol isnt mc.numCol then throw new UE.UserException( "'mb, mc''s dimension mismatch" )
    JC.matrixAdd( @, mb, mc )
    mc

  setZero: ->
    @elements.setValue(0)
    @


class MatrixBatchD
  constructor: ( m, n, matrices ) ->
# Read only properties
    @numRow = Math.ceil( m )
    @numCol = Math.ceil( n )
    @transposed = false
    @count = 0
    @MatrixDArray = []
    # !Read only properties

    # Private member
    @elementsArray = []
    @batchPointerArray = undefined
    # !Private member

    # Filter parameter
    if @numRow < 1 or @numCol < 1 then throw new UE.UserException( "'m, n' <uint32> must greater than zero" )
    if matrices instanceof Array and matrices.length > 0
      for m in matrices
        if !( m instanceof MatrixD ) then throw new UE.UserException( "'matrices' <MatrixD> array type mismatch" )
        # TODO: Support transposed matrix
        if m.transposed is true then throw new UE.UserException( "'matrices' <MatrixD> construct <MatrixBatchD> shouldn't be transposed" )
        if m.numRow isnt @numRow or m.numCol isnt @numCol then throw new UE.UserException( "'matrices''s dimension mismatch" )
        @MatrixDArray.push( m )
        @elementsArray.push( m.elements )
      @count = @elementsArray.length
      @batchPointerArray = new JC.BatchPointerArray( @elementsArray )


# Explicitly reclaim device memory , cudaFree under the hood
  destroy: ->
    @batchPointerArray.destroy()
    @batchPointerArray = undefined
    @numRow = 0
    @numCol = 0
    @count = 0
    @MatrixDArray = undefined
    @elementsArray = undefined
    undefined

  setZero: ->
    for m in @MatrixDArray
      m.setZero()
    @

  T: ->
    t = new MatrixBatchD( @numCol, @numRow )
    t.transposed = true
    t.count = @MatrixDArray.length
    t.MatrixDArray = @MatrixDArray
    t.elementsArray = @elementsArray
    t.batchPointerArray = @batchPointerArray
    t


  multiplyMatrixBatch: ( mbb, mcb ) ->
    if !( mbb instanceof MatrixBatchD and mcb instanceof MatrixBatchD ) then throw new UE.UserException( "'mbb, mcb''s  must be MatrixBatchD" )
    if @count isnt mbb.count isnt mcb.count then throw new UE.UserException( "'mbb, mcb''s dimension mismatch" )
    if @numCol isnt mbb.numRow or @numRow isnt mcb.numRow or mbb.numCol isnt mcb.numCol then throw new UE.UserException( "'mbb, mcb''s dimension mismatch" )
    JC.matrixMulMatrixBatched( @, mbb, mcb )
    mcb




module.exports = JC
exports = module.exports
exports.VectorD = VectorD
exports.MatrixD = MatrixD
exports.MatrixBatchD = MatrixBatchD