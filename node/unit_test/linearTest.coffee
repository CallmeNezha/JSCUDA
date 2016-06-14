assert = require( "assert" )
JC = require( "../linear" )


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

describe("Linear Functions Validation Test"
, ->
  it("vector add: ..."
  , () ->  #asynchronized test
    testLength = 1e6
    warmUpLength = 10


    # Warm up the device
    v1h = new Float32Array( ( Math.random() for num in [0...warmUpLength] ) )
    vAd = new JC.VectorD( v1h.length, v1h )
    v2h = new Float32Array( ( Math.random() for num in [0...warmUpLength] ) )
    gpuResult = new Float32Array( warmUpLength )
    vBd = new JC.VectorD( warmUpLength )
    # Device -> Host
    vBd.copyFrom( warmUpLength, v2h )
    # Measure time
    start = Date.now()
    vAd.add( vBd )
    console.log( "\tWarm up... JC #{ warmUpLength } elements vector add used:#{ Date.now() - start } ms" )
    # Host -> Device
    vAd.copyTo( warmUpLength, gpuResult )
    vAd.destroy()
    vBd.destroy()

    # GPU pass
    v1h = new Float32Array( ( Math.random() for num in [0...testLength] ) )
    vAd = new JC.VectorD( v1h.length, v1h )
    v2h = new Float32Array( ( Math.random() for num in [0...testLength] ) )
    gpuResult = new Float32Array( testLength )

    start = Date.now()
    vBd = new JC.VectorD( testLength )
    console.log( "\tJC #{ testLength } elements vector add used:#{ Date.now() - start } ms" )

    vBd.copyFrom( testLength, v2h )
    vAd.add( vBd )
    vAd.copyTo( testLength, gpuResult )
    vAd.destroy()
    vBd.destroy()

    # CPU pass
    start = Date.now()
    for e, i in v2h
      v1h[i] += e
    console.log( "\tV8 #{ testLength } elements vector add used:#{ Date.now() - start } ms" )

    # CPU & GPU comparison output
    meanError = 0
    for e, i in v1h
      meanError +=  Math.abs( e - gpuResult[i] )
    meanError /= v1h.length
    console.log( "\tv8.CPU. vs jc.GPU. Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754" )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )

  it("vector copy: ..."
  , () ->  #asynchronized test
    testLength = 1e6
    warmUpLength = 10


    # Warm up the device
    v1h = new Float32Array( ( Math.random() for num in [0...warmUpLength] ) )
    vAd = new JC.VectorD( v1h.length, v1h )
    v2h = new Float32Array( ( Math.random() for num in [0...warmUpLength] ) )
    gpuResult = new Float32Array( warmUpLength )
    vBd = new JC.VectorD( warmUpLength )
    # Device -> Host
    vBd.copyFrom( warmUpLength, v2h )
    # Measure time
    start = Date.now()
    vAd.copy( vBd )
    console.log( "\tWarm up... JC #{ warmUpLength } elements vector add used:#{ Date.now() - start } ms" )
    # Host -> Device
    vAd.copyTo( warmUpLength, gpuResult )
    vAd.destroy()
    vBd.destroy()

    # GPU pass
    v1h = new Float32Array( ( Math.random() for num in [0...testLength] ) )
    vAd = new JC.VectorD( v1h.length, v1h )
    v2h = new Float32Array( ( Math.random() for num in [0...testLength] ) )
    gpuResult = new Float32Array( testLength )

    start = Date.now()
    vBd = new JC.VectorD( testLength )
    console.log( "\tJC #{ testLength } elements vector add used:#{ Date.now() - start } ms" )

    vBd.copyFrom( testLength, v2h )
    vAd.copy( vBd )
    vAd.copyTo( testLength, gpuResult )
    vAd.destroy()
    vBd.destroy()

    # CPU pass
    start = Date.now()
    for e, i in v2h
      v1h[i] = e
    console.log( "\tV8 #{ testLength } elements vector add used:#{ Date.now() - start } ms" )

    # CPU & GPU comparison output
    meanError = 0
    for e, i in v1h
      meanError +=  Math.abs( e - gpuResult[i] )
    meanError /= v1h.length
    console.log( "\tv8.CPU. vs jc.GPU. Mean absolute error: #{ meanError } , \<float32\> refer to IEEE-754" )

    # If failed
    assert( AbsoluteError > meanError > -AbsoluteError, "test failed" )
  )
)





