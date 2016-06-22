assert = require( "assert" )
colors = require( "colors" )
JC = require( "../JC.js" )


colors.setTheme({
  warmup: 'black'
  cpu: 'cyan'
  gpu: 'green'
  standardIEEE: [ 'white', 'underline' ]
})


# Test parameters
AbsoluteError = 1e-6
JC.cudaDeviceInit()


dumpMatrix = (mat) ->
  for r in [0...mat.numRow]
    oneRow = "["
    for c in [0...mat.numCol]
      oneRow += mat.elements[c*mat.numRow + r] + " "
    oneRow += "]"
    console.log( "#{ oneRow }" )


describe("CUDA Base Utils Validation Check"
, ->
  @timeout( 0 ) # Disable mocha timeout for stress test

  it("CUDA: cudaInitDevice"
  , (done) ->  #asynchronized test
    assert.equal( typeof JC.cudaDeviceInit, 'function' )
    done()
  )
  it("CUDA: cudaDeviceReset"
  , (done) ->  #asynchronized test
    assert.equal( typeof JC.cudaDeviceReset, 'function' )
    done()
  )
)

describe("VectorD Functions Validation Test"
, ->
  it("vector add: ..."
  , () ->  #asynchronized test
    testLength = 1e7
    warmUpLength = 10

    # GPU pass
    for len in [warmUpLength, testLength]
      # Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len] ) )
      v2h = new Float32Array( ( Math.random() for num in [0...len] ) )
      gpuResult = new Float32Array( len )

      vAd = new JC.VectorD( v1h.length, v1h ) # Direct assign value to device memory
      vBd = new JC.VectorD( len )    # Reserve device memory
      # Device -> Host
      vBd.copyFrom( len, v2h ) # Copy value to device memory

      # Measure time
      start = Date.now()
      vAd.add( vBd )
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"

      # Device -> Host
      vAd.copyTo( len, gpuResult )
      mid = Date.now()
      vAd.copyTo( len, gpuResult )
      end = Date.now()
      console.log( "\t#{tip}JC(GPU) <<< VectorD::add >>> #{ len } elements used:#{ Math.max( mid - start - ( end - mid ), 0 ) } ms"[colorLog] )

      vAd.destroy()
      vBd.destroy()

    # CPU pass
    colorLog = "cpu"
    start = Date.now()
    for e, i in v2h
      v1h[i] += e
    console.log( "\tV8(CPU) <<< vector add >>> #{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )

    # CPU & GPU comparison output
    meanError = 0
    for e, i in v1h
      meanError +=  Math.abs( e - gpuResult[i] )
    meanError /= v1h.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("vector copy: ..."
  , () ->  #asynchronized test
    testLength = 1e7
    warmUpLength = 10

    # GPU pass
    for len in [testLength]
# Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len] ) )
      v2h = new Float32Array( ( Math.random() for num in [0...len] ) )
      gpuResult = new Float32Array( len )

      vAd = new JC.VectorD( v1h.length, v1h ) # Direct assign value to device memory
      vBd = new JC.VectorD( len )    # Reserve device memory
      # Device -> Host
      vBd.copyFrom( len, v2h ) # Copy value to device memory

      # Measure time
      start = Date.now()
      vAd.copy( vBd )
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"

      # Device -> Host
      vAd.copyTo( len, gpuResult )
      mid = Date.now()
      vAd.copyTo( len, gpuResult )
      end = Date.now()
      console.log( "\t#{tip}JC(GPU) <<< VectorD::copy >>> #{ len } elements used:#{ Math.max( mid - start - ( end - mid ), 0 ) } ms"[colorLog] )
      vAd.destroy()
      vBd.destroy()

    # CPU pass
    start = Date.now()
    for e, i in v2h
      v1h[i] = e
    colorLog = "cpu"
    console.log( "\t#{tip}V8(CPU) <<< vector copy >>> #{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )

    # CPU & GPU comparison output
    meanError = 0
    for e, i in v1h
      meanError +=  Math.abs( e - gpuResult[i] )
    meanError /= v1h.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("vector multiply scalar: ..."
  , () ->  #asynchronized test
    testLength = 1e7
    warmUpLength = 10

    # GPU pass
    scalar = new Float32Array( [Math.random()] )
    for len in [testLength]
# Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len] ) )
      gpuResult = new Float32Array( len )

      vAd = new JC.VectorD( v1h.length, v1h ) # Direct assign value to device memory

      # Measure time
      start = Date.now()
      vAd.multiplyScalar( scalar[0] )
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"

      # Device -> Host
      vAd.copyTo( len, gpuResult )
      mid = Date.now()
      vAd.copyTo( len, gpuResult )
      end = Date.now()
      console.log( "\t#{tip}JC(GPU) <<< VectorD::multiplyScalar >>> #{ len } elements used:#{ Math.max( mid - start - ( end - mid ), 0 ) } ms"[colorLog] )
      vAd.destroy()

    # CPU pass
    start = Date.now()
    for _, i in v1h
      v1h[i] *= scalar[0]
    colorLog = "cpu"
    console.log( "\t#{tip}V8(CPU) <<< vector multiply scalar >>> #{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )
    # CPU & GPU comparison output
    meanError = 0
    for e, i in v1h
      meanError +=  Math.abs( e - gpuResult[i] )
    meanError /= v1h.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("vector dot: ..."
  , () ->  #asynchronized test
    testLength = 1e7
    warmUpLength = 10

    # GPU pass
    for len in [testLength]
# Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len] ) )
      v2h = new Float32Array( ( Math.random() for num in [0...len] ) )

      vAd = new JC.VectorD( v1h.length, v1h ) # Direct assign value to device memory
      vBd = new JC.VectorD( len )    # Reserve device memory
      # Device -> Host
      vBd.copyFrom( len, v2h ) # Copy value to device memory

      # Measure time
      start = Date.now()
      gpuResult = vAd.dot( vBd )
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"
      console.log( "\t#{tip}JC(GPU) <<< VectorD::dot >>> #{ len } elements used:#{ Date.now() - start } ms"[colorLog] )

      vAd.destroy()
      vBd.destroy()

    # CPU pass
    start = Date.now()
    cpuresult = 0
    for i in [0...v1h.length]
      cpuresult += v1h[i]*v2h[i]
    colorLog = "cpu"
    console.log( "\t#{tip}V8(CPU) <<< vector dot >>> #{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )

    # CPU & GPU comparison output
    meanError = ( cpuresult - gpuResult ) / v1h.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("vector Euclidean norm: ..."
  , () ->  #asynchronized test
    testLength = 1e7
    warmUpLength = 10

    # GPU pass
    for len in [testLength]
# Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len] ) )

      vAd = new JC.VectorD( v1h.length, v1h ) # Direct assign value to device memory

      # Measure time
      start = Date.now()
      gpuResult = vAd.norm()
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"
      console.log( "\t#{tip}JC(GPU) <<< VectorD::norm >>> #{ len } elements used:#{ Date.now() - start } ms"[colorLog] )

      vAd.destroy()

    # CPU pass
    start = Date.now()
    cpuresult = 0
    for e in v1h
      cpuresult += e * e
    cpuresult = Math.sqrt(cpuresult)
    colorLog = "cpu"
    console.log( "\t#{tip}V8(CPU) <<< vector norm >>> #{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )

    # CPU & GPU comparison output
    meanError = ( cpuresult - gpuResult ) / v1h.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("vector tensor product: ..."
  , () ->  #asynchronized test
    testLength = 1e3
    warmUpLength = 10

    # GPU pass
    for len in [testLength]
# Warm up the device
      v1h = new Float32Array( ( Math.random() for num in [0...len] ) )
      v2h = new Float32Array( ( Math.random() for num in [0...len] ) )
      gpuResult = new Float32Array( v1h.length * v2h.length )
      vAd = new JC.VectorD( v1h.length, v1h ) # Direct assign value to device memory
      vBd = new JC.VectorD( v2h.length, v2h )

      matd = new JC.MatrixD( v1h.length, v2h.length )

      # Measure time
      start = Date.now()
      vAd.tensor( vBd, matd )
      if len is warmUpLength
        tip = "Warm up pass "
        colorLog = "warmup"
      else
        tip = ""
        colorLog = "gpu"

      # Device -> Host
      matd.copyTo( matd.numCol * matd.numRow, gpuResult  )
      mid = Date.now()
      matd.copyTo( matd.numCol * matd.numRow, gpuResult  )
      end = Date.now()
      console.log( "\t#{tip}JC(GPU) <<< VectorD::tensor >>> #{ len }x#{ len } elements used:#{ Math.max( mid - start - ( end - mid ), 0 ) } ms"[colorLog] )
      matd.destroy()
      vAd.destroy()
      vBd.destroy()

    # CPU pass
    cpuresult = new Float32Array( v1h.length * v2h.length )
    start = Date.now()
    # Column major matrix
    for v2e, j in v2h
      for v1e, i in v1h
        cpuresult[v1h.length * j + i] = v2e * v1e
    colorLog = "cpu"
    console.log( "\t#{tip}V8(CPU) <<< vector tensor product >>> #{ testLength }x#{ testLength } elements used:#{ Date.now() - start } ms"[colorLog] )

    # CPU & GPU comparison output
    meanError = 0
    for _, i in cpuresult
      meanError += cpuresult[i] - gpuResult[i]

    meanError /= cpuresult.length
    colorLog = "standardIEEE"
    console.log( "\tv8.CPU vs jc.GPU Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754"[colorLog] )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )
)







